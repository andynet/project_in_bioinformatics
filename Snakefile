configfile: "config.yaml"

wildcard_constraints:
    study="TCGA|GTEx",
    type="exon|gene",
    size="full|[0-9]+x[0-9]+",

rule help:
    shell:
        """
        echo "
********************************************************************************
        available entrytargets:
            get_description
            get_scaled
            get_filtered
********************************************************************************
        "
        """

rule get_description:
    input:
        expand("{data_dir}/{type}/{size}/{study}/raw/counts.desc",
                data_dir = config['data_dir'],
                study = config['study'],
                type = config['type'],
                size = config['size']
                ),

rule download:
    output:
        "{data_dir}/{type}/{study}/rse.Rdata",
    params:
        "{data_dir}/{type}/{study}",
    conda:
        "envs/recount.yaml",
    shell:
        """
        Rscript scripts/download.R              \
            --data_dir {wildcards.data_dir}     \
            --study {wildcards.study}           \
            --type {wildcards.type}

        mv {params}/rse_{wildcards.type}.Rdata {params}/rse.Rdata
        """

rule extract:
    input:
        "{data_dir}/{type}/{study}/rse.Rdata",
    output:
        "{data_dir}/{type}/{size}/{study}/raw/counts.tsv",
        "{data_dir}/{type}/{size}/{study}/raw/samples.tsv",
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

rule describe:
    input:
        "{data_dir}/{type}/{size}/{study}/raw/counts.tsv",
    output:
        "{data_dir}/{type}/{size}/{study}/raw/counts.desc",
    conda:
        "envs/py_data.yaml",
    shell:
        """
        python3 scripts/describe.py         \
            -i {input}                      \
            -o {output}
        """

rule get_scaled:
    input:
        expand("{data_dir}/{type}/{size}/{study}/scaled/counts.tsv",
                data_dir = config['data_dir'],
                study = config['study'],
                type = config['type'],
                size = config['size']
                ),

rule normalize:
    input:
        "{data_dir}/{type}/{size}/{study}/raw/counts.tsv",
    output:
        "{data_dir}/{type}/{size}/{study}/normalized/counts.tsv",
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
        "{data_dir}/{type}/{size}/{study}/normalized/counts.tsv",
    output:
        "{data_dir}/{type}/{size}/{study}/scaled/counts.tsv",
    conda:
        "envs/py_data.yaml",
    shell:
        """
        python3 scripts/scale.py            \
            -i {input}                      \
            -o {output}
        """

rule filter_naive:
    input:
        "{data_dir}/{type}/{size}/GTEx/scaled/counts.tsv",
        "{data_dir}/{type}/{size}/GTEx/raw/samples.tsv",
        "{data_dir}/{type}/{size}/TCGA/scaled/counts.tsv",
    output:
        "{data_dir}/{type}/{size}/filtered/naive/counts.tsv",
    conda:
        "envs/py_data.yaml",
    shell:
        """
        python3 scripts/filter_naive.py     \
            --gtex_counts {input[0]}        \
            --gtex_samples {input[1]}       \
            --tcga_counts {input[2]}        \
            --output {output}
        """

rule filter_pca:
    input:
        "{data_dir}/{type}/{size}/GTEx/scaled/counts.tsv",
        "{data_dir}/{type}/{size}/TCGA/scaled/counts.tsv",
    output:
        "{data_dir}/{type}/{size}/filtered/pca/counts.tsv",
    conda:
        "envs/py_data.yaml",
    shell:
        """
        python3 scripts/filter_pca.py       \
            --gtex_counts {input[0]}        \
            --tcga_counts {input[1]}        \
            --output {output}
        """

rule create_dummies:
    input:
        "{data_dir}/{type}/{size}/{study}/raw/samples.tsv",
    output:
        "{data_dir}/{type}/{size}/{study}/dummy/samples.tsv",
    conda:
        "envs/py_data.yaml",
    shell:
        """
        python3 scripts/create_dummies.py   \
            -i {input}                      \
            -o {output}
        """
# rule neural_network:
#     input:
#     output:
#     shell:
#         """
#         """
