# Data analysis for "Energy Efficiency of Speculative RAG" (Section 4.5).
#
#   Rscript analysis/analysis.R experiments/specrag_pubhealth/run_table.csv results/pubhealth
#
# Each benchmark is analysed separately. Packages: afex (RM-ANOVA), ARTool (ART
# ANOVA); everything else is base R. All tests are two-sided, alpha = 0.05,
# Holm-Bonferroni correction within each research question.

suppressPackageStartupMessages({
  library(afex)
  library(ARTool)
})

args <- commandArgs(trailingOnly = TRUE)
in_csv  <- if (length(args) >= 1) args[1] else "experiments/specrag_pubhealth/run_table.csv"
out_dir <- if (length(args) >= 2) args[2] else "results"
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)
ALPHA <- 0.05
sink(file.path(out_dir, "analysis_log.txt"), split = TRUE)

# ------------------------------------------------------------------ load data
raw <- read.csv(in_csv, stringsAsFactors = FALSE)
cat("Runs in table:", nrow(raw), " not ok:", sum(raw$status != "ok"), "\n")
d <- subset(raw, status == "ok" & !is.na(E_q_J))
cat("Runs used:", nrow(d), "\n")
TR  <- paste0("T", 0:9)
MK  <- data.frame(treatment = TR, m = c(NA, rep(c(2, 5, 10), each = 3)),
                  k = c(NA, rep(c(1, 2, 4), 3)))
d$treatment <- factor(d$treatment, levels = TR)

# ---------------------------------------------- measurement reliability (4.2)
cv <- aggregate(E_q_J ~ qid + treatment, d, function(x) sd(x) / mean(x))
names(cv)[3] <- "cv_E"
cat("\nCoefficient of variation of E_q across repetitions (per question x treatment):\n")
print(summary(cv$cv_E))
write.csv(cv, file.path(out_dir, "reliability_cv.csv"), row.names = FALSE)

# ----------------------------- average the repetitions of each question x treatment
num <- c("E_q_J", "E_gpu_J", "E_cpu_J", "T_q_s", "P_avg_W", "E_tok_J",
         "N_in", "N_int", "N_final", "N_gen", "correct", "correct_norm")
num <- num[num %in% names(d)]
for (v in num) d[[v]] <- suppressWarnings(as.numeric(d[[v]]))
a <- aggregate(d[, num], by = list(qid = d$qid, treatment = d$treatment), FUN = mean, na.rm = TRUE)
# greedy decoding + fixed seeds -> the answer is identical across repetitions;
# a question counts as correct if it was correct in the majority of repetitions
a$correct <- as.integer(a$correct >= 0.5)
a <- merge(a, MK, by = "treatment")
a$treatment <- factor(a$treatment, levels = TR)
# keep only questions that have all 10 treatments (complete blocks)
complete <- names(which(table(a$qid) == length(TR)))
a <- subset(a, qid %in% complete)
cat("Complete blocks (questions):", length(complete), "\n")
write.csv(a, file.path(out_dir, "per_question_treatment_means.csv"), row.names = FALSE)

# -------------------------------------------------------------- descriptives
desc <- do.call(rbind, lapply(TR, function(t) {
  x <- subset(a, treatment == t)
  data.frame(treatment = t,
             E_mean = mean(x$E_q_J), E_median = median(x$E_q_J), E_sd = sd(x$E_q_J),
             T_mean = mean(x$T_q_s), T_median = median(x$T_q_s), T_sd = sd(x$T_q_s),
             P_mean = mean(x$P_avg_W), P_median = median(x$P_avg_W), P_sd = sd(x$P_avg_W),
             Etok_mean = mean(x$E_tok_J, na.rm = TRUE), N_in_mean = mean(x$N_in),
             N_int_mean = mean(x$N_int), accuracy_pct = 100 * mean(x$correct))
}))
E0 <- desc$E_mean[desc$treatment == "T0"]
desc$E_diff_vs_T0_pct <- 100 * (desc$E_mean - E0) / E0
desc$kWh_per_1M_queries <- desc$E_mean * 1e6 / 3.6e6
cat("\nDescriptive statistics per treatment:\n")
print(format(desc, digits = 4), row.names = FALSE)
write.csv(desc, file.path(out_dir, "descriptives.csv"), row.names = FALSE)

png(file.path(out_dir, "boxplots.png"), width = 1500, height = 500, res = 110)
par(mfrow = c(1, 3))
boxplot(E_q_J ~ treatment, a, main = "Energy per query", ylab = "J", las = 2)
boxplot(T_q_s ~ treatment, a, main = "Execution time", ylab = "s", las = 2)
boxplot(P_avg_W ~ treatment, a, main = "Average power", ylab = "W", las = 2)
invisible(dev.off())

# ------------------------------------------------------------------ helpers
cliffs_delta <- function(x, y) {           # Cliff (1993); Romano et al. (2006) thresholds
  dom <- outer(x, y, "-")
  (sum(dom > 0) - sum(dom < 0)) / (length(x) * length(y))
}
paired_test <- function(x, y, label) {
  dif <- x - y
  sw  <- if (length(unique(dif)) > 2) shapiro.test(dif)$p.value else NA
  normal <- !is.na(sw) && sw > ALPHA
  if (normal) {
    tt <- t.test(x, y, paired = TRUE)
    data.frame(metric = label, test = "paired t-test", shapiro_p = sw,
               statistic = unname(tt$statistic), p = tt$p.value,
               effect = "Cohen's d_z", effect_size = mean(dif) / sd(dif))
  } else {
    wt <- wilcox.test(x, y, paired = TRUE, exact = FALSE)
    data.frame(metric = label, test = "Wilcoxon signed-rank", shapiro_p = sw,
               statistic = unname(wt$statistic), p = wt$p.value,
               effect = "Cliff's delta", effect_size = cliffs_delta(x, y))
  }
}

# -------------------------------------------------------------- RQ1: T0 vs T5
cat("\n=== RQ1: Standard RAG (T0) vs Speculative RAG m=5,k=2 (T5) ===\n")
w <- reshape(a[a$treatment %in% c("T0", "T5"), c("qid", "treatment", "E_q_J", "T_q_s", "P_avg_W", "correct")],
             idvar = "qid", timevar = "treatment", direction = "wide")
rq1 <- rbind(paired_test(w$E_q_J.T5, w$E_q_J.T0, "E_q"),
             paired_test(w$T_q_s.T5, w$T_q_s.T0, "T_q"),
             paired_test(w$P_avg_W.T5, w$P_avg_W.T0, "P_avg"))
# exact McNemar test = binomial test on the discordant pairs
b <- sum(w$correct.T0 == 1 & w$correct.T5 == 0); c <- sum(w$correct.T0 == 0 & w$correct.T5 == 1)
mc_p <- if (b + c > 0) binom.test(b, b + c, 0.5)$p.value else 1
rq1 <- rbind(rq1, data.frame(metric = "correctness", test = "exact McNemar", shapiro_p = NA,
                             statistic = b + c, p = mc_p, effect = "discordant (T0 only / T5 only)",
                             effect_size = NA))
rq1$p_holm <- p.adjust(rq1$p, "holm")
rq1$reject_H0 <- rq1$p_holm < ALPHA
print(rq1, row.names = FALSE)
cat(sprintf("Correct only under T0: %d, only under T5: %d\n", b, c))
write.csv(rq1, file.path(out_dir, "rq1_tests.csv"), row.names = FALSE)

# ------------------------------------------------ RQ2: m x k within Speculative RAG
cat("\n=== RQ2: effects of m and k on E_q (T1-T9) ===\n")
s <- subset(a, treatment != "T0")
s$m <- factor(s$m); s$k <- factor(s$k); s$qid <- factor(s$qid)
fit <- aov_ez(id = "qid", dv = "E_q_J", data = s, within = c("m", "k"),
              anova_table = list(correction = "GG", es = "pes"))
res <- residuals(lm(E_q_J ~ m * k + qid, data = s))
sw_res <- shapiro.test(res)$p.value
cat("Shapiro-Wilk on ANOVA residuals: p =", format(sw_res, digits = 4), "\n")
png(file.path(out_dir, "rq2_qq_residuals.png"), width = 600, height = 600, res = 110)
qqnorm(res); qqline(res); invisible(dev.off())
if (sw_res > ALPHA) {
  tab <- as.data.frame(fit$anova_table)
  rq2 <- data.frame(effect = rownames(tab), method = "RM-ANOVA (GG)", F = tab$F,
                    p = tab$`Pr(>F)`, partial_eta_sq = tab$pes)
} else {
  art_fit <- art(E_q_J ~ m * k + Error(qid), data = s)
  tab <- as.data.frame(anova(art_fit))
  rq2 <- data.frame(effect = as.character(tab$Term), method = "ART ANOVA", F = tab$`F value`,
                    p = tab$`Pr(>F)`, partial_eta_sq = NA)
}
rq2$p_holm <- p.adjust(rq2$p, "holm")
rq2$reject_H0 <- rq2$p_holm < ALPHA
print(rq2, row.names = FALSE)
write.csv(rq2, file.path(out_dir, "rq2_tests.csv"), row.names = FALSE)
png(file.path(out_dir, "rq2_interaction.png"), width = 700, height = 550, res = 110)
with(s, interaction.plot(k, m, E_q_J, fun = mean, ylab = "mean E_q (J)", xlab = "k (documents per draft)",
                         trace.label = "m (drafts)"))
invisible(dev.off())

# ------------------------------------------------- RQ3: time as a proxy for energy
cat("\n=== RQ3: execution time vs energy (all treatments) ===\n")
sp <- cor.test(a$T_q_s, a$E_q_J, method = "spearman", exact = FALSE)
fr <- friedman.test(P_avg_W ~ treatment | qid, data = a)
n_b <- length(unique(a$qid))
kendall_w <- unname(fr$statistic) / (n_b * (length(TR) - 1))
rq3 <- data.frame(hypothesis = c("H3a: rho(T_q, E_q) = 0", "H3b: equal P_avg across treatments"),
                  test = c("Spearman", "Friedman"),
                  statistic = c(unname(sp$estimate), unname(fr$statistic)),
                  p = c(sp$p.value, fr$p.value),
                  effect = c("rho", "Kendall's W"), effect_size = c(unname(sp$estimate), kendall_w))
rq3$p_holm <- p.adjust(rq3$p, "holm")
rq3$reject_H0 <- rq3$p_holm < ALPHA
print(rq3, row.names = FALSE)
write.csv(rq3, file.path(out_dir, "rq3_tests.csv"), row.names = FALSE)
# does ranking treatments by time give the same order as ranking by energy?
rank_cmp <- data.frame(treatment = desc$treatment, rank_by_T = rank(desc$T_mean), rank_by_E = rank(desc$E_mean))
cat("Kendall tau between treatment rankings by time and by energy:",
    format(cor(rank_cmp$rank_by_T, rank_cmp$rank_by_E, method = "kendall"), digits = 3), "\n")
write.csv(rank_cmp, file.path(out_dir, "rq3_rankings.csv"), row.names = FALSE)
png(file.path(out_dir, "rq3_time_vs_energy.png"), width = 700, height = 600, res = 110)
plot(a$T_q_s, a$E_q_J, col = as.integer(a$treatment), pch = 19, xlab = "T_q (s)", ylab = "E_q (J)")
legend("topleft", legend = TR, col = seq_along(TR), pch = 19, cex = 0.7, ncol = 2)
invisible(dev.off())

# --------------------------------------------------- energy / quality trade-off
cat("\n=== Energy vs answer quality ===\n")
tradeoff <- desc[, c("treatment", "E_mean", "E_diff_vs_T0_pct", "accuracy_pct")]
acc0 <- tradeoff$accuracy_pct[tradeoff$treatment == "T0"]
tradeoff$lower_energy_no_quality_loss <- tradeoff$E_mean < E0 & tradeoff$accuracy_pct >= acc0
print(format(tradeoff, digits = 4), row.names = FALSE)
write.csv(tradeoff, file.path(out_dir, "energy_quality_tradeoff.csv"), row.names = FALSE)
sink()
