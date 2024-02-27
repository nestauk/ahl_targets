library(here)
library(tidyverse)
library(scales)


year1 <- read_csv(here("outputs", "reports", "chart_csv", "one_year.csv") %>% str_remove("ahl_targets/analysis/bmi_modelling/"))

year2 <- read_csv(here("outputs", "reports", "chart_csv", "two_year.csv") %>% str_remove("ahl_targets/analysis/bmi_modelling/"))

year3 <- read_csv(here("outputs", "reports", "chart_csv", "three_year.csv") %>% str_remove("ahl_targets/analysis/bmi_modelling/"))

df <- rbind(year1 %>% mutate(year = 1),
            year2 %>% mutate(year = 2),
            year3 %>% mutate(year = 3)) %>%
  mutate(year = ifelse(variable == "baseline_prop", 0 , year))

df <- df [!duplicated(df[c(1,2,3)]),]

# obese only

ob_df <- df %>%
  filter(bmi_class %in% c("obese", "morbidly obese")) %>%
  group_by(variable, year) %>%
  summarise(value = sum(value)) %>%
  mutate(pp = round((value - 28.04679)/28.04679 * 100,2))

ggplot(ob_df, aes(x = year, y = value, group = year)) +
  geom_bar(stat = "identity", fill="steelblue") +
  geom_text(aes(label = label_percent()(value/100)), vjust = -0.3, size=3.5) +
  theme_minimal()

ggplot(ob_df, aes(x = year, y = pp, group = year)) +
  geom_bar(stat = "identity", fill="steelblue") +
  geom_text(aes(label = label_percent()(pp/100)), vjust = 1.3, size=3.5) +
  labs(title = "Obesity prevalence reduction by year",
       y = "%") +
  theme_minimal()
