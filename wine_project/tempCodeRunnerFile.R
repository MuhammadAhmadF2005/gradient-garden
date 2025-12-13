# ==============================================================================
# PROJECT: Dataset Preprocessing and Visualization in R
# COURSE: CS202 - ICT | GIKI
# DATASET: Wine Quality Dataset 
# ==============================================================================

# --- 0. Setup and Libraries ---
# Install packages if not already installed
if(!require(tidyverse)) install.packages("tidyverse")
if(!require(corrplot)) install.packages("corrplot")
if(!require(caret)) install.packages("caret")
if(!require(gridExtra)) install.packages("gridExtra")

library(tidyverse) # For data manipulation and plotting
library(corrplot)  # For correlation matrices
library(caret)     # For data splitting and preprocessing
library(gridExtra) # For arranging plots

# ==============================================================================
# STEP 1: Data Importing and Preprocessing [cite: 29]7
# ==============================================================================

# 1.1 Load the Dataset [cite: 30]
# We are using the Wine Quality dataset. Ensure the CSV is in your working directory.
# This dataset typically uses semi-colons ';' as separators.
wine_data <- read.csv("winequality-red.csv", sep = ";") 

# Check the structure
str(wine_data)

# 1.2 Data Cleaning: Missing Values [cite: 32]
# Checking for NA values
sum(is.na(wine_data))

# Handling missing values (Imputation Strategy)
# If NAs exist, we impute with the median for numerical columns.
# (This code block runs safely even if there are 0 NAs)
wine_data <- wine_data %>%
  mutate(across(where(is.numeric), ~ifelse(is.na(.), median(., na.rm = TRUE), .)))

# 1.3 Data Cleaning: Duplicate Rows [cite: 33]
# Count duplicates
num_duplicates <- sum(duplicated(wine_data))
print(paste("Number of duplicate rows removed:", num_duplicates))

# Remove duplicates
wine_data <- wine_data[!duplicated(wine_data), ]

# 1.4 Categorical Encoding [cite: 34]
# The target variable is 'quality' (score 3-8). 
# We will convert it into a Factor for classification (Good vs. Average/Poor).
# Let's assume quality >= 6 is 'Good' and < 6 is 'Not Good'.
wine_data$quality_label <- ifelse(wine_data$quality >= 6, "Good", "Bad")
wine_data$quality_label <- as.factor(wine_data$quality_label)

# ==============================================================================
# STEP 2: Exploratory Data Analysis (EDA) [cite: 35]
# ==============================================================================

# 2.1 Summary Statistics [cite: 36]
summary(wine_data)

# 2.2 Target Variable Analysis [cite: 37, 38]
# Visualizing the count distribution of the target variable (Quality Label)
ggplot(wine_data, aes(x = quality_label, fill = quality_label)) +
  geom_bar() +
  labs(title = "Distribution of Target Variable (Wine Quality)", 
       x = "Quality Class", y = "Count") +
  theme_minimal()

# 2.3 Feature Distribution (Numerical) [cite: 39, 40]
# Histogram for Alcohol content
p1 <- ggplot(wine_data, aes(x = alcohol)) +
  geom_histogram(binwidth = 0.5, fill = "skyblue", color = "black") +
  labs(title = "Distribution of Alcohol Content", x = "Alcohol", y = "Frequency")

# Histogram for pH
p2 <- ggplot(wine_data, aes(x = pH)) +
  geom_histogram(binwidth = 0.1, fill = "salmon", color = "black") +
  labs(title = "Distribution of pH", x = "pH", y = "Frequency")

grid.arrange(p1, p2, ncol = 2)

# ==============================================================================
# STEP 3: Data Visualization [cite: 42]
# ==============================================================================

# 3.1 Correlation Matrix [cite: 43]
# Select only numeric columns for correlation
numeric_vars <- wine_data %>% select_if(is.numeric)
cor_matrix <- cor(numeric_vars)

# Plotting the correlation matrix
corrplot(cor_matrix, method = "color", type = "upper", 
         tl.col = "black", tl.cex = 0.8, title = "Feature Correlation Matrix", mar=c(0,0,1,0))

# 3.2 Scatter Plots [cite: 44]
# visualising relationship between Fixed Acidity and Density
ggplot(wine_data, aes(x = fixed.acidity, y = density, color = quality_label)) +
  geom_point(alpha = 0.6) +
  labs(title = "Scatter Plot: Fixed Acidity vs Density", 
       x = "Fixed Acidity", y = "Density") +
  theme_minimal()

# 3.3 Boxplots for Target vs. Feature [cite: 45]
# Comparing Alcohol content across Wine Quality classes
ggplot(wine_data, aes(x = quality_label, y = alcohol, fill = quality_label)) +
  geom_boxplot() +
  labs(title = "Alcohol Content by Wine Quality", 
       x = "Quality Class", y = "Alcohol Content") +
  theme_minimal()

# ==============================================================================
# STEP 4: Feature Engineering [cite: 46]
# ==============================================================================

# 4.1 Create New Features [cite: 48, 79] (Creativity Point)
# Creating a 'total_acidity' feature by combining fixed and volatile acidity.
wine_data$total_acidity <- wine_data$fixed.acidity + wine_data$volatile.acidity

# 4.2 Feature Scaling [cite: 47]
# Normalizing numerical features (Min-Max Scaling) using Caret's preProcess
# We exclude the factor target variable 'quality_label'
preproc <- preProcess(wine_data[, -which(names(wine_data) == "quality_label")], method = c("range"))
wine_data_scaled <- predict(preproc, wine_data[, -which(names(wine_data) == "quality_label")])

# Add the target variable back
wine_data_scaled$quality_label <- wine_data$quality_label

# ==============================================================================
# STEP 5: Model Preparation [cite: 49]
# ==============================================================================

# 5.1 Split Data into Train and Test Sets [cite: 50]
# We use an 80/20 split
set.seed(123) # Ensure reproducibility
trainIndex <- createDataPartition(wine_data_scaled$quality_label, p = .8, 
                                  list = FALSE, 
                                  times = 1)

train_data <- wine_data_scaled[ trainIndex,]
test_data  <- wine_data_scaled[-trainIndex,]

# Outputting sizes to verify split
print(paste("Training set size:", nrow(train_data)))
print(paste("Testing set size:", nrow(test_data)))

# End of Script