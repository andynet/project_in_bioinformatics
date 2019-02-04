#!/home/andyb/miniconda3/envs/pib/bin/Rscript
# options(echo=TRUE) # if you want see commands in output file

library('recount')

args <- commandArgs(trailingOnly = TRUE)
print(args)

if (length(args) != 1) {
    print("Usage: \nextract.R GTEx \nextract.R TCGA")
    quit(save = "no")
}

name <- args[1]
types <- c('rse_gene', 'rse_exon')

for (type in types) {

    in_file <- paste(name, '/', type, '.Rdata', sep = '')

    if (type == 'rse_gene') {
        load(in_file)
        rse <- scale_counts(rse_gene)
    } else {
        load(in_file)
        rse <- scale_counts(rse_exon)
    }

    rse_subset <- rse[c(1:100), c(1:100)]

    # counts original
    counts <- data.frame(assays(rse)$counts)
    out_file <- paste(name, '/counts_', type, '.tsv', sep = '')
    write.table(counts, out_file, sep = '\t', col.names=NA)

    # counts subset
    counts_subset <- data.frame(assays(rse_subset)$counts)
    out_file <- paste(name, '/counts_subset_', type, '.tsv', sep = '')
    write.table(counts_subset, out_file, sep = '\t', col.names=NA)

    # source
    individuals <- data.frame(colData(rse))
    primary_site <- ifelse(name == 'TCGA', individuals['gdc_cases.project.primary_site'], individuals['smts'])
    out_file <- paste(name, '/individuals_', type, '.tsv', sep = '')
    write.table(primary_site, out_file, sep = '\t', col.names=NA)

    # source subset
    individuals <- data.frame(colData(rse_subset))
    primary_site <- ifelse(name == 'TCGA', individuals['gdc_cases.project.primary_site'], individuals['smts'])
    out_file <- paste(name, '/individuals_subset_', type, '.tsv', sep = '')
    write.table(primary_site, out_file, sep = '\t', col.names=NA)

}

