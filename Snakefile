configfile: "config.yaml"

wildcard_constraints:
    study="TCGA|GTEx",
    type="exon|gene",
    size="full|subset",

rule main:
    input:
        expand("{data_dir}/{study}/{size}/rse_{type}.df.tsv",
                data_dir = config['data_dir'],
                study = config['study'],
                type = config['type'],
                size = config['size']
                ),

rule download:
    output:
        "{data_dir}/{study}/rse_{type}.Rdata",
    conda:
        "envs/recount.yaml",
    shell:
        """
        Rscript scripts/download.R              \
            --data_dir {wildcards.data_dir}     \
            --study {wildcards.study}           \
            --type {wildcards.type}
        """

rule extract:
    input:
        "{data_dir}/{study}/rse_{type}.Rdata",
    output:
        "{data_dir}/{study}/{size}/rse_{type}.counts.tsv",
        "{data_dir}/{study}/{size}/rse_{type}.samples.tsv",
    conda:
        "envs/recount.yaml",
    shell:
        """
        Rscript scripts/extract.R               \
            --data_dir {wildcards.data_dir}     \
            --study {wildcards.study}           \
            --type {wildcards.type}             \
            --size {wildcards.size}
        """

rule normalize:
    input:
        "{data_dir}/{study}/{size}/rse_{type}.counts.tsv",
    output:
        "{data_dir}/{study}/{size}/rse_{type}.counts.norm.tsv",
    conda:
        "envs/py_data.yaml",
    shell:
        """
        python3 scripts/normalize.py        \
            -i {input}                      \
            -o {output}
        """

rule scale:
    input:
        "{data_dir}/{study}/{size}/rse_{type}.counts.norm.tsv",
    output:
        "{data_dir}/{study}/{size}/rse_{type}.counts.scaled.tsv",
    conda:
        "envs/py_data.yaml",
    shell:
        """
        python3 scripts/scale.py            \
            -i {input}                      \
            -o {output}
        """

rule filter:
    input:
        "{data_dir}/{study}/{size}/rse_{type}.counts.scaled.tsv",
        "{data_dir}/{study}/{size}/rse_{type}.samples.tsv",
    output:
        "{data_dir}/{study}/{size}/rse_{type}.df.tsv",
    conda:
        "envs/py_data.yaml",
    shell:
        """
        python3 scripts/filter.py           \
            -c {input[0]}                   \
            -s {input[1]}                   \
            -o {output}
        """

# rule neural_network:
#     input:
#     output:
#     shell:
#         """
#         """
