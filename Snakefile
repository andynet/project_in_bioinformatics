configfile: "config.yaml"

wildcard_constraints:
    type="exon|gene",
    size="full|subset",

rule main:
    input:
        expand("{data_dir}/{study}/rse_{type}.Rdata",
                data_dir = config['data_dir'],
                study = config['study'],
                type = config['type'],
                ),

rule download:
    output:
        "{data_dir}/{study}/rse_{type}.Rdata",
    conda:
        "envs/recount.yaml",
    shell:
        """
        Rscript scripts/download.R \
            {wildcards.data_dir} \
            {wildcards.study} \
            {wildcards.type}
        """

# rule extract:
#     input:
#         "{data_dir}/{study}/rse_{type}.Rdata",
#     output:
#         "{data_dir}/{study}/{size}/counts_rse_{type}.tsv",
#         "{data_dir}/{study}/{size}/samples_rse_{type}.tsv",
#     conda:
#         "envs/recount.yaml",
#     shell:
#         """
#         Rscript scripts/extract.R   \
#             {input[0]}
#         """
#
# rule normalize:
#     input:
#         "{data_dir}/{study}/{size}/counts_rse_{type}.tsv",
#     output:
#         "{data_dir}/{study}/{size}/counts_norm_rse_{type}.tsv",
#     shell:
#         """
#         # TODO: python script
#         """
#
# rule scale:
#     input:
#         "{data_dir}/{study}/{size}/counts_norm_rse_{type}.tsv",
#     output:
#         "{data_dir}/{study}/{size}/counts_norm_scaled_rse_{type}.tsv",
#     shell:
#         """
#         # TODO: python script
#         """
#
# rule filter:
#     input:
#         "",
#     output:
#         "",
#     shell:
#         """
#         # TODO: python script
#         """
#
# rule neural_network:
#     input:
#     output:
#     shell:
#         """
#         """
