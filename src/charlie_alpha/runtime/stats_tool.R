args <- commandArgs(trailingOnly = TRUE)
suppressPackageStartupMessages({
  library(jsonlite)
  library(survival)
  library(lme4)
  library(sandwich)
  library(lmtest)
})

request <- fromJSON(args[[1]], simplifyVector = FALSE)
frame <- read.csv(request$data_path, check.names = FALSE)
variables <- request$variables
method <- request$method_id
options <- request$analysis_options

if (!is.null(options$row_filters)) {
  for (filter_spec in options$row_filters) {
    column <- filter_spec$column
    values <- unlist(filter_spec$values)
    if (!(column %in% names(frame))) stop(paste("filter column does not exist:", column))
    if (filter_spec$operation == "include") {
      frame <- frame[frame[[column]] %in% values, , drop = FALSE]
    } else if (filter_spec$operation == "exclude") {
      frame <- frame[!(frame[[column]] %in% values), , drop = FALSE]
    } else {
      stop(paste("unsupported row filter operation:", filter_spec$operation))
    }
  }
}
if (!is.null(options$binary_recodes)) {
  for (recode in options$binary_recodes) {
    column <- recode$column
    positive <- unlist(recode$positive_values)
    if (!(column %in% names(frame))) stop(paste("recode column does not exist:", column))
    missing <- is.na(frame[[column]])
    frame[[column]] <- as.integer(frame[[column]] %in% positive)
    frame[[column]][missing] <- NA_integer_
  }
}
if (nrow(frame) == 0) stop("declared preprocessing removed every row")

require_column <- function(name) {
  value <- variables[[name]]
  if (is.null(value) || !(value %in% names(frame))) {
    stop(paste("missing or unknown variable role:", name))
  }
  value
}

result <- tryCatch({
  if (method == "wilcoxon_signed_rank") {
    before <- require_column("before")
    after <- require_column("after")
    fit <- wilcox.test(frame[[after]], frame[[before]], paired = TRUE, exact = FALSE)
    list(n_pairs = sum(complete.cases(frame[, c(before, after)])),
         statistic = unname(fit$statistic), p_value = fit$p.value)
  } else if (method == "firth_logistic") {
    if (!requireNamespace("brglm2", quietly = TRUE)) stop("brglm2 is unavailable")
    outcome <- require_column("outcome")
    predictors <- unlist(variables$predictors)
    if (!all(predictors %in% names(frame))) stop("unknown predictor")
    formula <- reformulate(predictors, response = outcome)
    fit <- glm(formula, data = frame, family = binomial("logit"),
               method = brglm2::brglmFit, type = "AS_mean")
    list(coefficients = as.list(coef(fit)), standard_errors = as.list(sqrt(diag(vcov(fit)))),
         p_values = as.list(coef(summary(fit))[, 4]))
  } else if (method == "negative_binomial_glm") {
    outcome <- require_column("outcome")
    predictors <- unlist(variables$predictors)
    formula <- reformulate(predictors, response = outcome)
    fit <- MASS::glm.nb(formula, data = frame)
    list(coefficients = as.list(coef(fit)), standard_errors = as.list(sqrt(diag(vcov(fit)))),
         theta = fit$theta, aic = AIC(fit))
  } else if (method == "mixed_effects") {
    outcome <- require_column("outcome")
    cluster <- require_column("cluster")
    predictors <- unlist(variables$predictors)
    formula <- as.formula(paste(outcome, "~", paste(predictors, collapse = "+"),
                                "+ (1|", cluster, ")"))
    fit <- lmer(formula, data = frame, REML = TRUE)
    standard_errors <- sqrt(diag(vcov(fit)))
    p_values <- 2 * pnorm(abs(fixef(fit) / standard_errors), lower.tail = FALSE)
    list(fixed_effects = as.list(fixef(fit)), standard_errors = as.list(standard_errors),
         p_values = as.list(p_values), random_effects = as.data.frame(VarCorr(fit)))
  } else if (method == "cox_ph") {
    time <- require_column("time")
    event <- require_column("event")
    predictors <- unlist(variables$predictors)
    formula <- as.formula(paste("Surv(", time, ",", event, ") ~",
                                paste(predictors, collapse = "+")))
    fit <- coxph(formula, data = frame, x = TRUE)
    proportional <- cox.zph(fit)
    list(coefficients = as.list(coef(fit)), hazard_ratios = as.list(exp(coef(fit))),
         standard_errors = as.list(sqrt(diag(vcov(fit)))),
         p_values = as.list(summary(fit)$coefficients[, "Pr(>|z|)"]),
         proportional_hazards_test = as.data.frame(proportional$table))
  } else if (method == "logrank") {
    time <- require_column("time")
    event <- require_column("event")
    group <- require_column("group")
    formula <- as.formula(paste("Surv(", time, ",", event, ") ~", group))
    fit <- survdiff(formula, data = frame)
    list(chisq = fit$chisq, degrees_of_freedom = length(fit$n) - 1,
         p_value = pchisq(fit$chisq, length(fit$n) - 1, lower.tail = FALSE),
         group_n = as.list(fit$n))
  } else {
    stop(paste("unsupported R method_id:", method))
  }
}, error = function(error) {
  list(.error = conditionMessage(error))
})

if (!is.null(result$.error)) {
  cat(toJSON(list(status = "error", error = result$.error), auto_unbox = TRUE), "\n")
  quit(status = 2)
}
cat(toJSON(list(status = "ok", result = result), auto_unbox = TRUE, null = "null"), "\n")
