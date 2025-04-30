# Load the dataset
data("USArrests")
View(USArrests)

# Hypothesis 1: Relationship between UrbanPop and Murder rates
# Correlation analysis
correlation <- cor(USArrests$UrbanPop, USArrests$Murder)
correlation

# Test the significance of the correlation
cor_test <- cor.test(USArrests$UrbanPop, USArrests$Murder)

# Output the correlation coefficient and test result
print(paste("Correlation coefficient:", correlation))
print(cor_test)

# Hypothesis 2: Difference in Assault rates between states with high and low UrbanPop percentages
# Categorize states into high and low UrbanPop groups
high_urbanpop <- USArrests$UrbanPop > mean(USArrests$UrbanPop)
low_urbanpop <- !high_urbanpop

# Perform a t-test to compare Assault rates between high and low UrbanPop groups
t_test_result <- t.test(USArrests$Assault[high_urbanpop], USArrests$Assault[low_urbanpop])

t_test_result1 <- t.test(USArrests$Assault[high_urbanpop], USArrests$Assault[low_urbanpop], var.equal = FALSE)

# Output the t-test result
print(t_test_result)
print(t_test_result1)

# Categorize states into high and low UrbanPop groups
high_urbanpop <- USArrests$UrbanPop > mean(USArrests$UrbanPop)
low_urbanpop <- !high_urbanpop

# Perform an F-test to compare variances of Assault rates between high and low UrbanPop groups
f_test_result <- var.test(USArrests$Assault[high_urbanpop], USArrests$Assault[low_urbanpop])

# Output the F-test result
print(f_test_result)
