# Assign the correct column names
names(RBC) <- c(
  "Rat_ID",
  "Parent_Box",
  "Density_per_um2",
  "Area_Fraction_percent",
  "Visual_Rank", "Annotations_Count"
)

# Remove the first row because it contains the old headers
RBC <- RBC[-1, ]

# Reset the row numbers
rownames(RBC) <- NULL

install.packages(c("ggplot2", "ggpubr"))
library(ggplot2)
library(ggpubr)

RBC[] <- lapply(RBC, function(x) {
  x <- trimws(as.character(x))
  x[x == ""] <- NA
  as.numeric(x)
})

RBC$Rat_ID <- as.integer(RBC$Rat_ID)
RBC$Parent_Box <- as.integer(RBC$Parent_Box)
RBC$Annotations_Count <- as.integer(RBC$Annotations_Count)

RBC$Density_per_um2 <- as.numeric(RBC$Density_per_um2)
RBC$Area_Fraction_percent <- as.numeric(RBC$Area_Fraction_percent)
RBC$Visual_Rank <- as.numeric(RBC$Visual_Rank)



### BOXPLOT OF PARENT 1 RANK VS DENSITY ### 

boxplot(
  Density_per_um2 ~ factor(Visual_Rank),
  data = RBC[
    RBC$Parent_Box == 1 &
      !is.na(RBC$Density_per_um2) &
      !is.na(RBC$Visual_Rank),
  ],
  xlab = "Visual blood clearance rank",
  ylab = "RBC count density per µm²",
  main = "Parent 1 RBC Density by Visual Rank"
)
stripchart(
  Density_per_um2 ~ factor(Visual_Rank),
  data = RBC[
    RBC$Parent_Box == 1 &
      !is.na(RBC$Density_per_um2) &
      !is.na(RBC$Visual_Rank),
  ],
  vertical = TRUE,
  method = "jitter",
  pch = 19,
  add = TRUE
)

### BOXPLOT OF PARENT 2 RANK VS DENSITY ###

boxplot(
  Density_per_um2 ~ factor(Visual_Rank),
  data = RBC[
    RBC$Parent_Box == 2 &
      !is.na(RBC$Density_per_um2) &
      !is.na(RBC$Visual_Rank),
  ],
  xlab = "Visual blood clearance rank",
  ylab = "RBC count density per µm²",
  main = "Parent 2 RBC Density by Visual Rank"
)
stripchart(
  Density_per_um2 ~ factor(Visual_Rank),
  data = RBC[
    RBC$Parent_Box == 2 &
      !is.na(RBC$Density_per_um2) &
      !is.na(RBC$Visual_Rank),
  ],
  vertical = TRUE,
  method = "jitter",
  pch = 19,
  add = TRUE
)


### BOXPLOT OF PARENT 1 AREA VS RANK ###

boxplot(
  Area_Fraction_percent ~ factor(Visual_Rank),
  data = RBC[
    RBC$Parent_Box == 1 &
      !is.na(RBC$Area_Fraction_percent) &
      !is.na(RBC$Visual_Rank),
  ],
  xlab = "Visual blood clearance rank",
  ylab = "RBC area fraction (%)",
  main = "Parent 1 RBC Area Fraction by Visual Rank"
)

stripchart(
  Area_Fraction_percent ~ factor(Visual_Rank),
  data = RBC[
    RBC$Parent_Box == 1 &
      !is.na(RBC$Area_Fraction_percent) &
      !is.na(RBC$Visual_Rank),
  ],
  vertical = TRUE,
  method = "jitter",
  pch = 19,
  add = TRUE
)

### BOXPLOT OF PARENT 2 AREA VS RANK ###

boxplot(
  Area_Fraction_percent ~ factor(Visual_Rank),
  data = RBC[
    RBC$Parent_Box == 2 &
      !is.na(RBC$Area_Fraction_percent) &
      !is.na(RBC$Visual_Rank),
  ],
  xlab = "Visual blood clearance rank",
  ylab = "RBC area fraction (%)",
  main = "Parent 2 RBC Area Fraction by Visual Rank"
)
stripchart(
  Area_Fraction_percent ~ factor(Visual_Rank),
  data = RBC[
    RBC$Parent_Box == 2 &
      !is.na(RBC$Area_Fraction_percent) &
      !is.na(RBC$Visual_Rank),
  ],
  vertical = TRUE,
  method = "jitter",
  pch = 19,
  add = TRUE
)

### BOXPLOT OF PARENT 1 ANNOTATIONS VS RANK ###

boxplot(
  Annotations_Count ~ factor(Visual_Rank),
  data = RBC[
    RBC$Parent_Box == 1 &
      !is.na(RBC$Annotations_Count) &
      !is.na(RBC$Visual_Rank),
  ],
  xlab = "Visual blood clearance rank",
  ylab = "RBC annotation count",
  main = "Parent 1 RBC Annotation Count by Visual Rank"
)

stripchart(
  Annotations_Count ~ factor(Visual_Rank),
  data = RBC[
    RBC$Parent_Box == 1 &
      !is.na(RBC$Annotations_Count) &
      !is.na(RBC$Visual_Rank),
  ],
  vertical = TRUE,
  method = "jitter",
  pch = 19,
  add = TRUE
)

### BOXPLOT OF PARENT 2 ANNOTATIONS VS RANK ###

boxplot(
  Annotations_Count ~ factor(Visual_Rank),
  data = RBC[
    RBC$Parent_Box == 2 &
      !is.na(RBC$Annotations_Count) &
      !is.na(RBC$Visual_Rank),
  ],
  xlab = "Visual blood clearance rank",
  ylab = "RBC annotation count",
  main = "Parent 2 RBC Annotation Count by Visual Rank"
)

stripchart(
  Annotations_Count ~ factor(Visual_Rank),
  data = RBC[
    RBC$Parent_Box == 2 &
      !is.na(RBC$Annotations_Count) &
      !is.na(RBC$Visual_Rank),
  ],
  vertical = TRUE,
  method = "jitter",
  pch = 19,
  add = TRUE
)


  #### Different ####


library(ggplot2)

# Keep complete observations
RBC_plot <- RBC[
  !is.na(RBC$Annotations_Count) &
    !is.na(RBC$Visual_Rank) &
    !is.na(RBC$Parent_Box),
]

# Make rank an ordered categorical variable
RBC_plot$Visual_Rank_Group <- factor(
  RBC_plot$Visual_Rank,
  levels = sort(unique(RBC_plot$Visual_Rank)),
  ordered = TRUE
)

# Label the parent boxes
RBC_plot$Parent_Box_Label <- factor(
  RBC_plot$Parent_Box,
  levels = c(1, 2),
  labels = c("Parent Box 1", "Parent Box 2")
)

ggplot(
  RBC_plot,
  aes(x = Visual_Rank_Group, y = Annotations_Count)
) +
  geom_boxplot(
    outlier.shape = NA
  ) +
  geom_jitter(
    width = 0.12,
    height = 0,
    size = 2,
    alpha = 0.7
  ) +
  stat_summary(
    fun = median,
    geom = "point",
    size = 3
  ) +
  stat_summary(
    aes(group = 1),
    fun = median,
    geom = "line",
    linewidth = 0.8
  ) +
  facet_wrap(~Parent_Box_Label) +
  labs(
    title = "RBC Annotation Count by Visual Blood Rank",
    x = "Visual blood clearance rank",
    y = "RBC annotation count"
  ) +
  theme_classic()

### 0 vs 3 comparison ###

# Keep only visual ranks 0 and 3
RBC_03 <- RBC[
  RBC$Visual_Rank %in% c(0, 3) &
    !is.na(RBC$Area_Fraction_percent),
]

# Make rank a two-level grouping variable
RBC_03$Rank_Group <- factor(
  RBC_03$Visual_Rank,
  levels = c(0, 3)
)

# Parent Box 1: rank 0 versus rank 3
P1_test <- wilcox.test(
  Area_Fraction_percent ~ Rank_Group,
  data = RBC_03[RBC_03$Parent_Box == 1, ],
  exact = FALSE
)

P1_test

# Parent Box 2: rank 0 versus rank 3
P2_test <- wilcox.test(
  Area_Fraction_percent ~ Rank_Group,
  data = RBC_03[RBC_03$Parent_Box == 2, ],
  exact = FALSE
)

P2_test


          ####### Trend Lines #######

library(ggplot2)

# Convert the three measurements into one long-format data frame
RBC_long <- rbind(
  data.frame(
    Rat_ID = RBC$Rat_ID,
    Parent_Box = RBC$Parent_Box,
    Visual_Rank = RBC$Visual_Rank,
    Measurement = "Annotation count",
    Value = RBC$Annotations_Count
  ),
  data.frame(
    Rat_ID = RBC$Rat_ID,
    Parent_Box = RBC$Parent_Box,
    Visual_Rank = RBC$Visual_Rank,
    Measurement = "Area fraction (%)",
    Value = RBC$Area_Fraction_percent
  ),
  data.frame(
    Rat_ID = RBC$Rat_ID,
    Parent_Box = RBC$Parent_Box,
    Visual_Rank = RBC$Visual_Rank,
    Measurement = "Count density per µm²",
    Value = RBC$Density_per_um2
  )
)

# Remove any incomplete observations
RBC_long <- RBC_long[
  complete.cases(RBC_long[, c("Parent_Box", "Visual_Rank", "Value")]),
]

# Create clearer parent labels
RBC_long$Parent <- factor(
  RBC_long$Parent_Box,
  levels = c(1, 2),
  labels = c("Parent Box 1", "Parent Box 2")
)

# Control the order of the rows
RBC_long$Measurement <- factor(
  RBC_long$Measurement,
  levels = c(
    "Annotation count",
    "Area fraction (%)",
    "Count density per µm²"
  )
)

# Make all six plots
ggplot(
  RBC_long,
  aes(x = Visual_Rank, y = Value)
) +
  geom_jitter(
    width = 0.04,
    height = 0,
    size = 2,
    alpha = 0.7
  ) +
  geom_smooth(
    method = "lm",
    formula = y ~ x,
    se = TRUE
  ) +
  facet_grid(
    Measurement ~ Parent,
    scales = "free_y"
  ) +
  scale_x_continuous(
    breaks = sort(unique(RBC_long$Visual_Rank))
  ) +
  labs(
    title = "QuPath Measurements Compared with Visual Clearance Rank",
    x = "Visual blood clearance rank",
    y = NULL
  ) +
  theme_classic()
    
  ## Colors ##

ggplot(RBC_long,aes(x = Visual_Rank, y = Value)) +geom_jitter(width = 0.04,height = 0,size = 2,alpha = 0.6) +
  geom_smooth(aes(color = Measurement,fill = Measurement),method = "lm",formula = y ~ x,se = TRUE,
    linewidth = 1) +facet_grid(Measurement ~ Parent,scales = "free_y") +scale_x_continuous(
    breaks = sort(unique(RBC_long$Visual_Rank))) +scale_color_manual(values = c(
      "Annotation count" = "#0072B2",
      "Area fraction (%)" = "#D55E00",
      "Count density per µm²" = "#009E73")) +
  scale_fill_manual(values = c(
      "Annotation count" = "#0072B2",
      "Area fraction (%)" = "#D55E00",
      "Count density per µm²" = "#009E73")) +
  labs(title = "QuPath Measurements Compared with Visual Clearance Rank",x = "Visual blood clearance rank",
    y = NULL,color = "Measurement",fill = "Measurement") +theme_classic()


            ######## Excluding Outliers #######


RBC_long_graph <- RBC_long[
  !RBC_long$Rat_ID %in% c(6, 9),
]
ggplot(
  RBC_long_graph,
  aes(x = Visual_Rank, y = Value)
) +
  geom_jitter(
    width = 0.04,
    height = 0,
    size = 2,
    alpha = 0.6
  ) +
  geom_smooth(
    aes(
      color = Measurement,
      fill = Measurement
    ),
    method = "lm",
    formula = y ~ x,
    se = TRUE,
    linewidth = 1
  ) +
  facet_grid(
    Measurement ~ Parent,
    scales = "free_y"
  ) +
  scale_x_continuous(
    breaks = sort(unique(RBC_long_graph$Visual_Rank))
  ) +
  scale_color_manual(
    values = c(
      "Annotation count" = "#0072B2",
      "Area fraction (%)" = "#D55E00",
      "Count density per µm²" = "#009E73"
    )
  ) +
  scale_fill_manual(
    values = c(
      "Annotation count" = "#0072B2",
      "Area fraction (%)" = "#D55E00",
      "Count density per µm²" = "#009E73"
    )
  ) +
  labs(
    title = "QuPath Measurements Compared with Visual Clearance Rank",
    subtitle = "Outliers excluded",
    x = "Visual blood clearance rank",
    y = NULL,
    color = "Measurement",
    fill = "Measurement"
  ) +
  theme_classic()



    ####### Correlation tests #######

##  Parent 1  ##

cor.test(
  RBC$Visual_Rank[RBC$Parent_Box == 1],
  RBC$Annotations_Count[RBC$Parent_Box == 1],
  method = "spearman",
  exact = FALSE
)


cor.test(
  RBC$Visual_Rank[RBC$Parent_Box == 1],
  RBC$Area_Fraction_percent[RBC$Parent_Box == 1],
  method = "spearman",
  exact = FALSE
)



cor.test(
  RBC$Visual_Rank[RBC$Parent_Box == 1],
  RBC$Density_per_um2[RBC$Parent_Box == 1],
  method = "spearman",
  exact = FALSE
)



##  Parent 2 ##

cor.test(
  RBC$Visual_Rank[RBC$Parent_Box == 2],
  RBC$Annotations_Count[RBC$Parent_Box == 2],
  method = "spearman",
  exact = FALSE
)


cor.test(
  RBC$Visual_Rank[RBC$Parent_Box == 2],
  RBC$Area_Fraction_percent[RBC$Parent_Box == 2],
  method = "spearman",
  exact = FALSE
)


cor.test(
  RBC$Visual_Rank[RBC$Parent_Box == 2],
  RBC$Density_per_um2[RBC$Parent_Box == 2],
  method = "spearman",
  exact = FALSE
)




 ### Add to the Figure ###


library(ggplot2)

# Function to format p-values nicely
format_p <- function(p) {
  paste0("p = ", format.pval(p, digits = 3, eps = 0))
}

# Create one label per panel
split_data <- split(
  RBC_long,
  list(RBC_long$Parent, RBC_long$Measurement),
  drop = TRUE
)


label_list <- lapply(split_data, function(d) {
  d <- d[!is.na(d$Visual_Rank) & !is.na(d$Value), ]
  
  test <- cor.test(
    d$Visual_Rank,
    d$Value,
    method = "spearman",
    exact = FALSE
  )
  
  data.frame(
    Parent = d$Parent[1],
    Measurement = d$Measurement[1],
    label = paste0(
      "\u03C1 = ", round(unname(test$estimate), 3),
      "\n", format_p(test$p.value)
    )
  )
})

label_df <- do.call(rbind, label_list)

# Add x/y positions for the labels
label_df$Visual_Rank <- Inf
label_df$Value <- Inf

# Plot
ggplot(
  RBC_long,
  aes(x = Visual_Rank, y = Value)
) +
  geom_jitter(
    width = 0.04,
    height = 0,
    size = 2,
    alpha = 0.6
  ) +
  geom_smooth(
    aes(color = Measurement, fill = Measurement),
    method = "lm",
    formula = y ~ x,
    se = TRUE,
    linewidth = 1
  ) +
  geom_text(
    data = label_df,
    aes(x = Visual_Rank, y = Value, label = label),
    inherit.aes = FALSE,
    hjust = 1.1,
    vjust = 1.2,
    size = 3.5
  ) +
  facet_grid(
    Measurement ~ Parent,
    scales = "free_y"
  ) +
  scale_x_continuous(
    breaks = sort(unique(RBC_long$Visual_Rank))
  ) +
  scale_color_manual(
    values = c(
      "Annotation count" = "#0072B2",
      "Area fraction (%)" = "#D55E00",
      "Count density per µm²" = "#009E73"
    )
  ) +
  scale_fill_manual(
    values = c(
      "Annotation count" = "#0072B2",
      "Area fraction (%)" = "#D55E00",
      "Count density per µm²" = "#009E73"
    )
  ) +
  labs(x = "Visual blood clearance grade",
    y = NULL,
    color = "Measurement",
    fill = "Measurement"
  ) +
  theme_classic()


