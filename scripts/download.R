#!/home/andyb/miniconda3/envs/pib/bin/Rscript
# options(echo=TRUE) # if you want see commands in output file

library('recount')

args <- commandArgs(trailingOnly = TRUE)
print(args)

if (length(args) != 2) {
    print("Usage: \ndownload.R SRP012682 GTEx \ndownload.R TCGA TCGA")
    quit(save = "no")
}

accession <- args[1]    # SRP012682 or TCGA
name <- args[2]     # GTEx or TCGA

# download
types <- c("rse-gene", "rse-exon")

for (type in types) {
    download_study(accession, type = type, name)
    Sys.sleep(3)
}
