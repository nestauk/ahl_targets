# install required packages (see below for names)

library(here)
library(tidyverse)
library(bw)
library(survey)
library(reshape2)
library(magrittr)


#Create base date frame from processed hse data
df <- read_csv(here("inputs/processed/hse_2019.csv") %>% str_remove("ahl_targets/analysis/bmi_modelling/")) %>%
  mutate(sex = ifelse(sex == 1, "male", "female"))

calculate_final_weight <- function(df, bmi_class_values, intake_diff_values) {

  # Filter data for overweight individuals
  dfover <- df %>%
    filter(bmi_class %in% bmi_class_values) %>%
    mutate(intake_diff = intake_diff_values)

  # Create a matrix with daily energy intake changes over 3 years
  eichange_3 <- t(apply(dfover, 1, function(x) rep(as.numeric(x["intake_diff"]), 365*3)))

  # Calculate final body weight using the provided function 'adult_weight'
  model_weight_3 <- adult_weight(dfover$weight, dfover$height/100, dfover$age, dfover$sex, eichange_3, days = 365*3)

  # Extract the final body weight list and convert it into a dataframe
  bw_3 <- model_weight_3$Body_Weight %>% as.data.frame() %>% select(V1095) %>% rename(final_weight = V1095)

  # Add the final body weight column to the original dataset
  dfover <- cbind(dfover, bw_3)

  dfunder <- df %>%
    filter(bmi <25) %>% #create data set with just Underweight + normal weight data
    mutate(intake_diff = intake_diff_values) %>% #calorie intake doesn't change for this group
    mutate(final_weight = weight)

  out <- rbind(dfunder, dfover)

  out$final_bmi <- out$final_weight / ((out$height/100)^2)

  ##Create final_bmi_class column
  out <- out %>%
    mutate(final_bmi_class = case_when(final_bmi <= 18.5 ~ "underweight",
                                       final_bmi > 18.5 & final_bmi < 25 ~ "normal",
                                       final_bmi >= 25 & final_bmi < 30 ~ "overweight",
                                       final_bmi >= 30 & final_bmi < 40 ~ "obese",
                                       final_bmi >= 40 ~ "morbidly obese",
                                       TRUE ~ "NA"))

  return(out)
}


values <- c(seq(0, -220, by = -1) , seq(-220, -500, by = -10))

intake_diff_values_list <- lapply(values, function(x) c(x))
num_vectors <- length(intake_diff_values_list)
bmi_class_values_list <- lapply(1:num_vectors, function(x) c("overweight", "obese", "morbidly obese"))


result_list <- mapply(calculate_final_weight, df = list(df), bmi_class_values = bmi_class_values_list, intake_diff_values = intake_diff_values_list, SIMPLIFY = FALSE)


result <- do.call("rbind", result_list) %>%
  mutate(bmi_class_r = ifelse(final_bmi_class == "morbidly obese", "obese", final_bmi_class ))


intake_diff = seq(-500,0,1)

perc_diff = intake_diff*1/432

test_df <- data.frame(intake_diff, perc_diff)


plot <- ggplot() +
  # Plot the first line using data from "test_df"
  geom_line(data = test_df, aes(x = intake_diff, y = perc_diff, color = "Linear", linetype = "Linear")) +

  # Plot the second line using data from "result"
  geom_line(data = result %>%
              group_by(intake_diff, final_bmi_class) %>%
              tally(wt = wt_int) %>%
              group_by(intake_diff) %>%
              mutate(perc = 100 * n / sum(n)) %>%
              filter(final_bmi_class == "obese") %>%
              mutate(perc_diff = (perc - 24.7)/24.7),
            aes(x = intake_diff, y = perc_diff, color = "Quadratic", linetype = "Quadratic")) +

  geom_hline(yintercept = c(0, -0.5)) +
  geom_vline(xintercept = c(-78, -216)) +

  labs(x = "Intake difference",
       y = "Obesity prevalence difference (%)",
       color = "Model",
       linetype = "Model") +


  theme_bw()

print(plot)


to_merge <- test_df %>% mutate(linear = perc_diff) %>% select(intake_diff, linear)


result %>%
  group_by(intake_diff, final_bmi_class) %>%
  tally(wt = wt_int) %>%
  group_by(intake_diff) %>%
  mutate(perc = 100 * n / sum(n)) %>%
  filter(final_bmi_class == "obese") %>%
  mutate(perc_diff = (perc - 24.69028)/24.69028) %>%
  merge(., to_merge, by = "intake_diff") %T>%
  write_csv(here("outputs", "reports", "chart_csv", "chartI.csv") %>% str_remove("ahl_targets/analysis/bmi_modelling/"))
