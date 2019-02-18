library(recount, quietly = TRUE)
library(argparser, quietly = TRUE)

# Create a parser
p <- arg_parser("Extract data from .Rdata file")

# Add command line arguments
p <- add_argument(p, "--data_dir", help = "data dir")
p <- add_argument(p, "--study", help = "TCGA|GTEx")
p <- add_argument(p, "--type", help = "gene|exon")
p <- add_argument(p, "--size", help = "full|subset")

# Parse the command line arguments
args <- c('--data_dir', '/home/andrej/ActiveProjects/project_in_bioinformatics/data', 
          '--study', 'GTEx',
          '--type', 'gene',
          '--size', 'subset')
argv <- parse_args(p, argv = args)

rdata_file <- paste(argv$data_dir, '/', argv$study, '/rse_', argv$type, '.Rdata', sep = '')
load(rdata_file)

subset_size <- 100

if (argv$type == "gene" & argv$size == "subset") {
    rse <- scale_counts(rse_gene[1:subset_size, 1:subset_size])
} else if (argv$type == "gene" & argv$size == "full") {
    rse <- scale_counts(rse_gene)
} else if (argv$type == "exon" & argv$size == "subset") {
    rse <- scale_counts(rse_exon[1:subset_size, 1:subset_size])
} else if (argv$type == "exon" & argv$size == "full") {
    rse <- scale_counts(rse_exon)
} else {
    print("Unexpected branch.")
    quit(save = "no", status = 1)
}

out_dir <- paste(argv$data_dir, '/', argv$study, '/', argv$size, sep = '')
# dir.create(out_dir)

counts <- data.frame(assays(rse)$counts)
out_file <- paste(out_dir, '/rse_', argv$type, '.counts.tsv', sep = '')
write.table(counts, out_file, sep = '\t', col.names=NA)


if (argv$study == "GTEx") {
    samples <- data.frame(colData(rse))['smts']
} else if (argv$study == "TCGA") {
    samples <- data.frame(colData(rse))['gdc_cases.project.primary_site']
} else {
    print("Unexpected branch.")
    quit(save = "no", status = 1)
}

out_file <- paste(out_dir, '/rse_', argv$type, '.samples.tsv', sep = '')
write.table(samples, out_file, sep = '\t', col.names=NA)
