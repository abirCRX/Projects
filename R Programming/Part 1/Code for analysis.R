library(readr)
library(dplyr)
library(corrplot)
library(ggplot2)
#rm(list=ls(all=TRUE))

pwtdata=read_csv("D:/tiss/RprogrammingHimanshusir/FINAL ASSIGNMENT/abir/Part 1/pwt90.csv")

View(pwtdata)
pwt2014data = pwtdata %>% filter(year == 2014)
View(pwt2014data)

projectdata <- select(pwt2014data, country, rgdpe, rgdpo, emp, labsh, hc, ctfp, cwtfp)
View(projectdata)
totalmissingvalues <- sum(is.na(projectdata))
totalmissingvalues
omiteddata <- na.omit(projectdata)
View(omiteddata)

summary(omiteddata)
correlationmatrix <- cor(omiteddata[2:8])
correlationmatrix
corrplot(correlationmatrix, method = "color", type = "full", order = "hclust", tl.col = "black", tl.srt = 45, addCoef.col = "black")

high_correlation_indices <- which(correlationmatrix == max(correlationmatrix, na.rm = TRUE), arr.ind = TRUE)
low_correlation_indices <- which(correlationmatrix == min(correlationmatrix, na.rm = TRUE), arr.ind = TRUE)

high_correlation_indices
low_correlation_indices

variable_names <- colnames(correlationmatrix)
cat("Variables with highest correlation:", variable_names[high_correlation_indices[1, 1]], "and", variable_names[high_correlation_indices[1, 2]], "\n")
cat("Variables with lowest correlation:", variable_names[low_correlation_indices[1, 1]], "and", variable_names[low_correlation_indices[1, 2]], "\n")


ggplot(omiteddata, aes(x = ctfp, y = labsh)) +
  geom_point() +
  geom_smooth(method = "lm", se = FALSE) +
  labs(x = "CTFP", y = "Labor Share", title = "Labor Share vs CTFP")


ggplot(omiteddata, aes(x = cwtfp, y = labsh)) +
  geom_point() +
  geom_smooth(method = "lm", se = FALSE) +
  labs(x = "Welfare Related TFP", y = "Labor Share", title = "Labor Share vs Welfare Related TFP")


ggplot(omiteddata, aes(x = hc, y = labsh)) +
  geom_point() +
  geom_smooth(method = "lm", se = FALSE) +
  labs(x = "Human Capital Index", y = "Labor Share", title = "Labor Share vs Human Capital Index")


ggplot(omiteddata, aes(x = ctfp, y = cwtfp)) +
  geom_point() +
  geom_smooth(method = "lm", se = FALSE) +
  labs(x = "CTFP", y = "Welfare Related TFP", title = "Welfare Related TFP vs CTFP")


ggplot(omiteddata, aes(x = hc, y = cwtfp)) +
  geom_point() +
  geom_smooth(method = "lm", se = FALSE) +
  labs(x = "Human Capital Index", y = "Welfare Related TFP", title = "Welfare Related TFP vs Human Capital Index")



labshctfp.lm <- lm(labsh ~ ctfp, omiteddata);
plot(omiteddata$labsh,omiteddata$ctfp);
abline(lm(labshctfp.lm));
summary(labshctfp.lm)


labshcwtfp.lm <- lm(labsh ~ cwtfp, omiteddata);
plot(omiteddata$labsh,omiteddata$cwtfp);
abline(lm(labshcwtfp.lm));
summary(labshcwtfp.lm)

labshhc.lm <- lm(labsh ~ hc, omiteddata);
plot(omiteddata$labsh,omiteddata$hc);
abline(lm(labshhc.lm));
summary(labshhc.lm)

cwtfpctfp.lm <- lm(cwtfp ~ ctfp, omiteddata);
plot(omiteddata$cwtfp,omiteddata$ctfp);
abline(lm(cwtfpctfp.lm));
summary(cwtfpctfp.lm)

cwtfphc.lm <- lm(cwtfp ~ hc, omiteddata);
plot(omiteddata$cwtfp,omiteddata$hc);
abline(lm(cwtfphc.lm));
summary(cwtfphc.lm)

rgdpocwtfp.lm <- lm(rgdpo ~ cwtfp, omiteddata);
plot(omiteddata$rgdpo,omiteddata$cwtfp);
abline(lm(rgdpocwtfp.lm));
summary(rgdpocwtfp.lm)

rgdpoctfp.lm <- lm(rgdpo ~ ctfp, omiteddata);
plot(omiteddata$rgdpo,omiteddata$ctfp);
abline(lm(rgdpoctfp.lm));
summary(rgdpoctfp.lm)



