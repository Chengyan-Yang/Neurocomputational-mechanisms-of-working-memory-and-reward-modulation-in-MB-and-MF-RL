# Generic posterior summary helper. It is sample-size agnostic; 52subs/64subs
# are handled by whichever fitted object is loaded before calling this function.
samplesSummary <- function(samples) {
  results <- c(mean=mean(samples),
               CI=quantile(samples, 0.025),
               CI=quantile(samples, 0.975),
               `% < 0`=sum(samples<0)/length(samples),
               `% > 0`=sum(samples>0)/length(samples))
  print(results, digits=4)
  invisible(results)
}
