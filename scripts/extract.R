library(recount, quietly = TRUE)
library(argparser, quietly = TRUE)

# Create a parser
p <- arg_parser("Extract data from .Rdata file")

# Add command line arguments
p <- add_argument(p, "--data_dir", help = "data dir")
p <- add_argument(p, "--study", help = "TCGA|GTEx")
p <- add_argument(p, "--type", help = "gene|exon")
p <- add_argument(p, "--size", help = "full|[0-9]+x[0-9]+")

# Parse the command line arguments
# argv <- parse_args(p, argv = c("--data_dir", "../data", 
#                                "--study", "GTEx", 
#                                "--type", "gene",
#                                "--size", "200x100"))
argv <- parse_args(p)

# "{data_dir}/{type}/{study}/rse.Rdata"
rdata_file <- paste(argv$data_dir, argv$type, argv$study, 'rse.Rdata', sep = '/')
print(rdata_file)
load(rdata_file)

if (argv$size == "full") {
    x_size <- dim(rse_gene)[1]
    y_size <- dim(rse_gene)[2]
} else if (grepl("^[0-9]+x[0-9]+$", argv$size)) {
    m <- regexpr("^[0-9]+", argv$size, perl = TRUE)
    x_size <- as.integer(regmatches(argv$size, m))
    m <- regexpr("[0-9]+$", argv$size, perl = TRUE)
    y_size <- as.integer(regmatches(argv$size, m))
} else {
    print("Unexpected size.")
    quit(save = "no", status = 1)
}

if (argv$type == "gene") {
    rse <- read_counts(rse_gene[1:x_size, 1:y_size])
} else if (argv$type == "exon") {
    rse <- read_counts(rse_exon[1:x_size, 1:y_size])
} else {
    print("Unexpected type.")
    quit(save = "no", status = 1)
}

# "{data_dir}/{type}/{size}/{study}/raw/counts.tsv."
out_dir <- paste(argv$data_dir, argv$type, argv$size, argv$study, "raw", sep = '/')
dir.create(out_dir, recursive = TRUE)

counts <- data.frame(assays(rse)$counts, check.names=FALSE)
out_file <- paste(out_dir, "counts.tsv", sep = '/')
write.table(counts, out_file, sep = '\t', col.names=NA)


if (argv$study == "GTEx") {
    samples <- data.frame(colData(rse))['smts']
} else if (argv$study == "TCGA") {
    samples <- data.frame(colData(rse))['gdc_cases.project.primary_site']
} else {
    print("Unexpected branch.")
    quit(save = "no", status = 1)
}

out_file <- paste(out_dir, "samples.tsv", sep = '/')
write.table(samples, out_file, sep = '\t', col.names=NA)
