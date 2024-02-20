library(here)
library(tidyverse)
library(bw)
library(survey)
library(reshape2)
library(magrittr)
library(psych)

##############################################################################################################################
###
####  ROBUSTNESS CHECKS ####
###
##############################################################################################################################


#Create base date frame from processed hse data
df <- read_csv(here("inputs/processed/hse_2019.csv") %>%
                 str_remove("ahl_targets/analysis/bmi_modelling/")) %>%
  mutate(sex = ifelse(sex == 1, "male", "female"))


bootstrap_func <- function(over_intake,obese_intake,mobese_intake){

  df_1_2 <- df %>%
    filter(bmi <25) %>% #create data set with just Underweight + normal weight data
    mutate(intake_diff = 0) %>% #calorie intake doesn't change for this group
    mutate(final_weight = weight)  #body weight doesn't change for this group

  df_3 <- df %>%
    filter(bmi >= 25, bmi < 30) %>% #create data set with just overweight people
    mutate(intake_diff = over_intake) #create new column for kcal reduction

  eichange_3 <- t(apply(df_3, 1, function(x) rep(as.numeric(x["intake_diff"]), 365*3))) #apply energy intake change over 3 years

  model_weight_3 <- adult_weight(df_3$weight, df_3$height/100, df_3$age, df_3$sex, eichange_3, days=365*3) #calculate final body weight

  bw_3 <- model_weight_3$Body_Weight %>% as.data.frame() %>% select(V1095) %>% rename(final_weight = V1095) #convert final body weight list into dataframe

  df_3 <- cbind(df_3, bw_3)

  df_4 <- df %>%
    filter(bmi >= 30, bmi < 40) %>%
    mutate(intake_diff = obese_intake)

  eichange_4 <- t(apply(df_4, 1, function(x) rep(as.numeric(x["intake_diff"]), 365*3)))

  model_weight_4 <- adult_weight(df_4$weight, df_4$height/100, df_4$age, df_4$sex, eichange_4, days=365*3)

  bw_4 <- model_weight_4$Body_Weight %>% as.data.frame() %>% select(V1095) %>% rename(final_weight = V1095)

  df_4 <- cbind(df_4, bw_4)

  df_5 <- df %>%
    filter(bmi >= 40) %>%
    mutate(intake_diff = mobese_intake)

  eichange_5 <- t(apply(df_5, 1, function(x) rep(as.numeric(x["intake_diff"]), 365*3)))

  model_weight_5 <- adult_weight(df_5$weight, df_5$height/100, df_5$age, df_5$sex, eichange_5, days=365*3)

  bw_5 <- model_weight_5$Body_Weight %>% as.data.frame() %>% select(V1095) %>% rename(final_weight = V1095)

  df_5 <- cbind(df_5, bw_5)

  df_complete <- rbind(df_1_2, df_3, df_4, df_5)

  ##Calculate final bmi
  df_complete$final_bmi <- df_complete$final_weight / ((df_complete$height/100)^2)

  ##Create final_bmi_class column
  df_complete <- df_complete %>%
    mutate(final_bmi_class = case_when(final_bmi <= 18.5 ~ "underweight",
                                       final_bmi > 18.5 & final_bmi < 25 ~ "normal",
                                       final_bmi >= 25 & final_bmi < 30 ~ "overweight",
                                       final_bmi >= 30 & final_bmi < 40 ~ "obese",
                                       final_bmi >= 40 ~ "morbidly obese",
                                       TRUE ~ "NA"))

  svydes <-  svydesign(ids=~df_complete$psu,
                       nest = T,
                       data=df_complete,
                       weights=df_complete$wt_int) #apply survey weights

  prop <- prop.table(svytable(~bmi_class, svydes)) %>%
    as.data.frame() %>%
    mutate(bmi_class = factor(bmi_class, levels = c("underweight", "normal", "overweight", "obese", "morbidly obese"))) %>%
    rename(baseline_prop = Freq)

  prop_final <- prop.table(svytable(~final_bmi_class, svydes)) %>%
    as.data.frame() %>%
    mutate(final_bmi_class = factor(final_bmi_class, levels = c("underweight", "normal", "overweight", "obese", "morbidly obese"))) %>%
    rename(final_prop = Freq)

  return(prop_final)

}



results_all <- list()

# Set the seed outside the loop to ensure reproducibility
set.seed(42)

# Number of evaluations
num_evaluations <- 1

deviation <- 0

for (i in 1:num_evaluations) {
  # Generate random values for intake_change_low and intake_change_high
  intake_change_pop <- -53.8

  scaling_factor <- runif(1, 0.64240331*(1-deviation), 0.64240331*(1 + deviation))

  scaled_intake <- intake_change_pop/scaling_factor

  over_scale <- runif(1, 0.95339*(1-deviation), 0.95339*(1 + deviation))
  obese_scale <- runif(1, 1.03878*(1-deviation), 1.03878*(1 + deviation))
  mobese_scale <- runif(1, 1.21504*(1-deviation), 1.21504*(1 + deviation))

  over_intake <- over_scale * scaled_intake
  obese_intake <- obese_scale * scaled_intake
  mobese_intake <- mobese_scale * scaled_intake

  # Call the overweight_func function
  result <- bootstrap_func(over_intake,obese_intake,mobese_intake)

  result <- result %>% mutate(pop_intake = intake_change_pop)

  # Append the result to the list
  results_all[[i]] <- result
}

result_df <- do.call("rbind", results_all)


over_df <- result_df %>% filter(final_bmi_class == "overweight")

mean(over_df$final_prop)


sd(over_df$final_prop)

quantile(over_df$final_prop)


ggplot(over_df, aes(x = final_prop)) +
  geom_histogram(bins = 10) +
  labs(title = "Overweight group",
       x = "Prevalence in the population") +
  theme_bw()

obese_df <- result_df %>% filter(final_bmi_class == "obese")

mean(obese_df$final_prop)


sd(obese_df$final_prop)

quantile(obese_df$final_prop)

ggplot(obese_df, aes(x = final_prop)) +
  geom_histogram(bins = 10) +
  labs(title = "Obese group",
       x = "Prevalence in the population") +
  theme_bw()

mobese_df <- result_df %>% filter(final_bmi_class == "morbidly obese")

mean(mobese_df$final_prop)


sd(mobese_df$final_prop)

quantile(mobese_df$final_prop)

ggplot(mobese_df, aes(x = final_prop)) +
  geom_histogram(bins = 5) +
  labs(title = "Morbidly obese group",
       x = "Prevalence in the population") +
  theme_bw()

total_obesity <- result_df %>%
  filter(final_bmi_class %in% c("morbidly obese", "obese")) %>%
  group_by(pop_intake) %>%
  summarise(final_prop = sum(final_prop)) %>%
  mutate(final_bmi_class = "total_obese")

mean(total_obesity$final_prop)


sd(total_obesity$final_prop)

quantile(total_obesity$final_prop)

ggplot(total_obesity, aes(x = final_prop)) +
  geom_histogram(bins = 5) +
  labs(title = "Total obese group",
       x = "Prevalence in the population") +
  theme_bw()


df_regroup <- df %>% mutate(bmi_class = ifelse(bmi_class %in% c("morbidly obese", "obese"), "obese", bmi_class))

svydes <-  svydesign(ids=~df_regroup$psu,
                     nest = T,
                     data=df_regroup,
                     weights=df_regroup$wt_int) #apply survey weights

prop <- prop.table(svytable(~bmi_class, svydes)) %>%
  as.data.frame() %>%
  mutate(bmi_class = factor(bmi_class, levels = c("underweight", "normal", "overweight", "obese"))) %>%
  rename(baseline_prop = Freq)

(mean(total_obesity$final_prop) - prop$baseline_prop[2])/prop$baseline_prop[2]

(round(mean(total_obesity$final_prop)*100 - prop$baseline_prop[2]*100,0))/round(prop$baseline_prop[2]*100,0)

(round(quantile(total_obesity$final_prop)[2]*100 - prop$baseline_prop[2]*100,0))/round(prop$baseline_prop[2]*100,0)

(round(quantile(total_obesity$final_prop)[4]*100 - prop$baseline_prop[2]*100,0))/round(prop$baseline_prop[2]*100,0)
