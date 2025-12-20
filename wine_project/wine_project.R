# ==============================================================================
# Dataset Preprocessing and Visualization in R
# Course: CS202 – ICT
# Dataset: Wine Quality (Red Wine)
# ==============================================================================

# ----------------------------
# Libraries
# ----------------------------

library(tidyverse)
library(corrplot)
library(caret)
library(gridExtra)

# ----------------------------
# Step 1: Data Import & Cleaning
# ----------------------------

# NOTE: Ensure 'sep' matches your file. If your file uses commas, change ";" to ","
wine_data <- read.csv("C:/Users/Lenovo/Downloads/wine.csv", sep = ",") 

# Check if data loaded correctly (if ncol is 1, the separator is wrong)
if(ncol(wine_data) <= 1) {
  warning("Data seems to have only 1 column. Check your 'sep' argument in read.csv")
}

# Handle missing values (median imputation for numeric features)
wine_data <- wine_data %>%
  mutate(across(where(is.numeric),
                ~ ifelse(is.na(.), median(., na.rm = TRUE), .)))

# Remove duplicate rows (Using distinct() is safer than base R here)
wine_data <- wine_data %>% distinct()

# Create categorical target variable
# We verify 'quality' exists before accessing it
if("quality" %in% names(wine_data)) {
  wine_data$quality_label <- ifelse(wine_data$quality >= 6, "Good", "Bad")
  wine_data$quality_label <- as.factor(wine_data$quality_label)
} else {
  stop("Column 'quality' not found. Please check your CSV load step.")
}

# ----------------------------
# Step 2: Exploratory Data Analysis
# ----------------------------

# Summary statistics
summary(wine_data)

# Target variable distribution
ggplot(wine_data, aes(x = quality_label, fill = quality_label)) +
  geom_bar() +
  labs(title = "Wine Quality Distribution",
       x = "Quality Class", y = "Count") +
  theme_minimal()

# Feature distributions
p1 <- ggplot(wine_data, aes(x = alcohol)) +
  geom_histogram(binwidth = 0.5, fill = "skyblue", color = "black") +
  labs(title = "Alcohol Distribution", x = "Alcohol", y = "Frequency")

p2 <- ggplot(wine_data, aes(x = pH)) +
  geom_histogram(binwidth = 0.1, fill = "salmon", color = "black") +
  labs(title = "pH Distribution", x = "pH", y = "Frequency")

grid.arrange(p1, p2, ncol = 2)

# ----------------------------
# Step 3: Data Visualization
# ----------------------------

# Correlation matrix
numeric_vars <- wine_data %>% select(where(is.numeric))
cor_matrix <- cor(numeric_vars)

corrplot(cor_matrix,
         method = "color",
         type = "upper",
         tl.col = "black",
         tl.cex = 0.8,
         title = "Correlation Matrix",
         mar = c(0, 0, 1, 0))

# Scatter plot
ggplot(wine_data, aes(x = fixed.acidity, y = density, color = quality_label)) +
  geom_point(alpha = 0.6) +
  labs(title = "Fixed Acidity vs Density",
       x = "Fixed Acidity", y = "Density") +
  theme_minimal()

# Boxplot
ggplot(wine_data, aes(x = quality_label, y = alcohol, fill = quality_label)) +
  geom_boxplot() +
  labs(title = "Alcohol Content by Wine Quality",
       x = "Quality Class", y = "Alcohol") +
  theme_minimal()

# ----------------------------
# Step 4: Feature Engineering
# ----------------------------

# New feature
wine_data$total_acidity <- wine_data$fixed.acidity + wine_data$volatile.acidity

# Feature scaling (Min-Max normalization)
preproc <- preProcess(wine_data %>% select(-quality_label),
                      method = "range")

wine_data_scaled <- predict(preproc, wine_data %>% select(-quality_label))
wine_data_scaled$quality_label <- wine_data$quality_label

# ----------------------------
# Step 5: Train-Test Split
# ----------------------------

set.seed(123)

trainIndex <- createDataPartition(wine_data_scaled$quality_label,
                                  p = 0.8,
                                  list = FALSE)

train_data <- wine_data_scaled[trainIndex, ]
test_data  <- wine_data_scaled[-trainIndex, ]

cat("Training set size:", nrow(train_data), "\n")
cat("Testing set size:", nrow(test_data), "\n")

