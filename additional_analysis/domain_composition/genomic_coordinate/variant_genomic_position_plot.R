library(ggplot2)
library(ggridges)
library(dplyr)
library(viridis)

# ---------- Read in variant dataframe ----------
read_variant_file <- function(path, label) {
  df <- read.delim(path, header = TRUE, sep = "\t")[, 1:2]
  colnames(df) <- c("chr", "pos")
  df$pos <- as.numeric(df$pos)
  df$type <- label
  return(df)
}


# ---------- 🔈 The path of variant file, some of the data cannot be re-distribution, please contact me if you want to use them ---------- 
# benign
# df_syn       <- read_variant_file("spatial_vis/synonymous.bn.txt", "Synonymous")
# df_missense  <- read_variant_file("spatial_vis/missense.bn.txt", "Missense")
# df_noncoding <- read_variant_file("spatial_vis/noncoding.bn.txt", "Noncoding")

# pathogenic
# df_syn       <- read_variant_file("spatial_vis/synonymous.pt.txt", "Synonymous")
# df_missense  <- read_variant_file("spatial_vis/missense.pt.txt", "Missense")
# df_noncoding <- read_variant_file("spatial_vis/noncoding.pt.txt", "Noncoding")

# all
df_syn       <- read_variant_file("spatial_vis/synonymous.txt", "Synonymous")
df_missense  <- read_variant_file("spatial_vis/missense.txt", "Missense")
df_noncoding <- read_variant_file("spatial_vis/noncoding.txt", "Noncoding")
df_all <- bind_rows(df_syn, df_missense, df_noncoding)


# ---------- GRCh38 chromosome length ----------
chrom_sizes <- data.frame(
  chr = paste0("chr", c(1:22, "X", "Y")),
  start = 1,
  end = c(
    248956422,242193529,198295559,190214555,181538259,170805979,
    159345973,145138636,138394717,133797422,135086622,133275309,
    114364328,107043718,101991189,90338345,83257441,80373285,
    58617616,64444167,46709983,50818468,156040895,57227415
  )
)


# ---------- join data ----------
df_plot <- df_all %>%
  inner_join(chrom_sizes, by = "chr") %>%
  mutate(
    rel_pos = pos / end,
    chr = factor(chr, levels = paste0("chr", c(1:22, "X", "Y")))
  )


# ---------- draw the pic ----------
p <- ggplot(df_plot, aes(x = rel_pos, y = chr, fill = type)) +
  geom_density_ridges(
    scale = 3, rel_min_height = 0.01, alpha = 0.7,
    color = scales::alpha("gray40", 0.4), linewidth = 0.2
  ) +
  scale_fill_manual(
    name = "Variant type",
    values = c(
      "Synonymous" = "#d3d3d3", # 
      "Missense"   = "#c71e1d", # #c71e1d #c55138
      "Noncoding"  = "#18a1cd" # #18a1cd #4b8599
      # "Splicing"   = "#15607a"
    )
  ) +
  theme_ridges(font_size = 13, grid = TRUE) +
  theme(
    # remove y grid
    panel.grid.major.x = element_blank(),
    panel.grid.minor.x = element_blank(),

    # set the font pattern of axis and title
    axis.text.y = element_text(size = 10, face = "plain", colour = "black"),
    axis.text.x = element_text(size = 10, face = "plain", colour = "black"),
    axis.title.y = element_text(size = 13, face = "plain"),
    axis.title.x = element_text(size = 13, face = "plain"),
    plot.title  = element_text(size = 14, face = "plain", hjust = 0.5),

    # prettier legend
    legend.position = "bottom",
    legend.title = element_text(size = 12, face = "plain"),
    legend.text = element_text(size = 11, face = "plain"),
    legend.key.height = unit(0.5, "cm"),
    legend.key.width  = unit(0.5, "cm"),
    legend.box = "horizontal",
    legend.direction = "horizontal",
    legend.spacing.x = unit(0.1, "cm"),
    legend.background = element_blank(),
    legend.box.margin = margin(t = -5),
    legend.justification = "center",
    panel.spacing = unit(0.15, "lines")
  ) +
  guides(fill = guide_legend(ncol = 4)) +
  labs(
    x = "Relative position along chromosome",
    y = "",
    title = "Genomic distribution of different variant types"
  )


# ---------- create and save ----------
dir.create("./plot_result", recursive = TRUE, showWarnings = FALSE)
# bengin
# ggsave("./plot_result/variant_distribution.bn.pdf", plot = p, width = 6.5, height = 7, dpi = 300)
# ggsave("./plot_result/variant_distribution.bn.jpg", plot = p, width = 6.5, height = 7, dpi = 300)

# patho
# ggsave("./plot_result/variant_distribution.pt.pdf", plot = p, width = 6.5, height = 7, dpi = 300)
# ggsave("./plot_result/variant_distribution.pt.jpg", plot = p, width = 6.5, height = 7, dpi = 300)

# all
ggsave("./plot_result/variant_distribution.pdf", plot = p, width = 6.5, height = 7, dpi = 300)
ggsave("./plot_result/variant_distribution.jpg", plot = p, width = 6.5, height = 7, dpi = 300)