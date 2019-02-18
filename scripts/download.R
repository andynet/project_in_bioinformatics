library(recount, quietly = TRUE)
library(argparser, quietly=TRUE)

# Create a parser
p <- arg_parser("Download recount data")

# Add command line arguments
p <- add_argument(p, "--data_dir", help = "data dir")
p <- add_argument(p, "--study", help = "TCGA|GTEx")
p <- add_argument(p, "--type", help = "gene|exon")

# Parse the command line arguments
argv <- parse_args(p)

if (argv$study == "TCGA") {
    accession <- "TCGA"
} else if (argv$study == "GTEx") {
    accession <- "SRP012682"
} else {
    print("Unexpected branch. Only studies TCGA and GTEx are supported.")
    quit(save = "no", status = 1)
}

if (argv$type == "gene") {
    type = "rse-gene"
} else if (argv$type == "exon") {
    type = "rse-exon"
} else {
    print("Unexpected branch. Only types gene and exon are supported.")
    quit(save = "no", status = 1)
}

outdir <- paste(argv$data_dir, '/', argv$study, sep = "")
download_study(project = accession, type = type, outdir = outdir)
