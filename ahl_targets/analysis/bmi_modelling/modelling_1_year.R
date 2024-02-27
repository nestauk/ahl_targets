# install required packages (see below for names)


library(here)
library(tidyverse)
library(bw)
library(survey)
library(reshape2)
library(magrittr)

##############################################################################################################################
###
####  OBESITY MODELLING ####
###
##############################################################################################################################


#Create base date frame from processed hse data
df <- read_csv(here("inputs/processed/hse_2019.csv") %>%
                 str_remove("ahl_targets/analysis/bmi_modelling/")) %>%
  mutate(sex = ifelse(sex == 1, "male", "female"))

## Split base data frame into underweight (class1) +normal (class2), overweight (class3), obese (class4) & severely obese (class5).
##Apply body weight change functions based on agreed calorie reductions in each bmi_class and create final body weight column

#Underweight + healthy weight data set (class 1 and 2)
df_1_2 <- df %>%
  filter(bmi <25) %>% #create data set with just Underweight + normal weight data
  mutate(intake_diff = 0) %>% #calorie intake doesn't change for this group
  mutate(final_weight = weight)  #body weight doesn't change for this group



#Overweight (class 3)
df_3 <- df %>%
  filter(bmi >= 25, bmi < 30) %>% #create data set with just overweight people
  mutate(intake_diff = -74.2) #create new column for kcal reduction

eichange_3 <- t(apply(df_3, 1, function(x) rep(as.numeric(x["intake_diff"]), 365))) #apply energy intake change over 3 years

model_weight_3 <- adult_weight(df_3$weight, df_3$height/100, df_3$age, df_3$sex, eichange_3, days=365) #calculate final body weight

bw_3 <- model_weight_3$Body_Weight %>% as.data.frame() %>% select(V365) %>% rename(final_weight = V365) #convert final body weight list into dataframe

df_3 <- cbind(df_3, bw_3) #bind final body weight column with original dataset


#Obese (class 4)
df_4 <- df %>%
  filter(bmi >= 30, bmi < 40) %>%
  mutate(intake_diff = -80.9)

eichange_4 <- t(apply(df_4, 1, function(x) rep(as.numeric(x["intake_diff"]), 365)))

model_weight_4 <- adult_weight(df_4$weight, df_4$height/100, df_4$age, df_4$sex, eichange_4, days=365)

bw_4 <- model_weight_4$Body_Weight %>% as.data.frame() %>% select(V365) %>% rename(final_weight = V365)

df_4 <- cbind(df_4, bw_4)


#Morbidly Obese (class 5)

df_5 <- df %>%
  filter(bmi >= 40) %>%
  mutate(intake_diff = -94.6)

eichange_5 <- t(apply(df_5, 1, function(x) rep(as.numeric(x["intake_diff"]), 365)))

model_weight_5 <- adult_weight(df_5$weight, df_5$height/100, df_5$age, df_5$sex, eichange_5, days=365)
bw_5 <- model_weight_5$Body_Weight %>% as.data.frame() %>% select(V365) %>% rename(final_weight = V365)

df_5 <- cbind(df_5, bw_5)


##Bind all the datasets together again
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

##Create summary table of BASELINE bmi_class distribution

svydes <-  svydesign(ids=~df_complete$psu,
                     nest = T,
                     data=df_complete,
                     weights=df_complete$wt_int) #apply survey weights

prop <- prop.table(svytable(~bmi_class, svydes)) %>%
  as.data.frame() %>%
  mutate(bmi_class = factor(bmi_class, levels = c("underweight", "normal", "overweight", "obese", "morbidly obese"))) %>%
  rename(baseline_prop = Freq) #create table

##Create summary table of FINAL bmi_class distribution

prop_final <- prop.table(svytable(~final_bmi_class, svydes)) %>%
  as.data.frame() %>%
  mutate(final_bmi_class = factor(final_bmi_class, levels = c("underweight", "normal", "overweight", "obese", "morbidly obese"))) %>%
  rename(final_prop = Freq)

#Create full bmi_class distribution table
bmi_class_dist <- cbind(prop, prop_final$final_prop) #combine summary tables of baseline and final bmi_class distributions

colnames(bmi_class_dist)[3] <- "final_prop" #change column names

bmi_class_dist <- bmi_class_dist %>%
  arrange(factor(bmi_class, levels = c("underweight", "normal", "overweight", "obese", "morbidly obese"))) %>% #rearrange row order
  mutate(baseline_prop = baseline_prop*100) %>% #multiply proportion values by 100
  mutate(final_prop = final_prop*100 ) #multiply proportion values by 100

bmi_class_dist %>%
  melt() %T>%
  write_csv(here("outputs", "reports", "chart_csv", "one_year.csv") %>% str_remove("ahl_targets/analysis/bmi_modelling/"))


bmi_class_dist %>%
  melt() %>%
  ggplot(., aes(x = bmi_class, y = value, fill = variable)) +
  geom_bar(stat = "identity", position = position_dodge(width=0.9)) +
  geom_text(aes(label = as.factor(round(value,1))), vjust= -0.5 , color = "black", position = position_dodge(width=0.9)) +
  labs(x = "BMI class",
       y = "Prevalence") +
  theme(axis.text.x = element_text(angle=90, vjust=.5, hjust=1))
