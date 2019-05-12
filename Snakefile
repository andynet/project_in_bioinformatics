configfile: "config.yaml"

wildcard_constraints:
    study="TCGA|GTEx",
    type="exon|gene",
    size="full|[0-9]+x[0-9]+",
    filter="pca|naive",
    architecture="[0-9-]+"

rule help:
    shell:
        """
        echo "
********************************************************************************
        available entrytargets:
            get_description
            get_scaled
            get_filtered
            run_nn
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
        "{data_dir}/{type}/{size}/{study}/raw/stages.tsv",
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
        "{data_dir}/{type}/{size}/naive/counts.tsv",
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
        "{data_dir}/{type}/{size}/pca/counts.tsv",
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

rule split_datasets:
    input:
        "{data_dir}/{type}/{size}/{filter}/counts.tsv",
        "{data_dir}/{type}/{size}/TCGA/dummy/samples.tsv",
    output:
        expand("{{data_dir}}/{{type}}/{{size}}/{{filter}}/counts.{set}.tsv",
                set = ['training', 'validation', 'testing']),
        expand("{{data_dir}}/{{type}}/{{size}}/{{filter}}/samples.{set}.tsv",
                set = ['training', 'validation', 'testing']),
    conda:
        "envs/py_data.yaml",
    shell:
        """
        python3 scripts/split_datasets.py   \
            --inf {input[0]}                \
            --inl {input[1]}                \
            --trf {output[0]}               \
            --vaf {output[1]}               \
            --tef {output[2]}               \
            --trl {output[3]}               \
            --val {output[4]}               \
            --tel {output[5]}
        """


rule nn_feedforward:
    input:
        "{data_dir}/{type}/{size}/{filter}/counts.training.tsv",
        "{data_dir}/{type}/{size}/{filter}/samples.training.tsv",
        "{data_dir}/{type}/{size}/{filter}/counts.validation.tsv",
        "{data_dir}/{type}/{size}/{filter}/samples.validation.tsv",
    output:
        "{data_dir}/{type}/{size}/{filter}/nn_{architecture}/loss.tsv",
    params:
        "{data_dir}/{type}/{size}/{filter}/nn_{architecture}/models",
        "{data_dir}/{type}/{size}/{filter}/nn_{architecture}/predictions",
        config['max_training_time'],
    conda:
        "envs/py_data.yaml",
    shell:
        """
        mkdir -p {params[0]}
        mkdir -p {params[1]}

        python3 scripts/nn_general.py               \
            --counts_t {input[0]}                   \
            --samples_t {input[1]}                  \
            --counts_v {input[2]}                   \
            --samples_v {input[3]}                  \
            --architecture {wildcards.architecture} \
            --loss {output[0]}                      \
            --model_dir {params[0]}                 \
            --prediction_dir {params[1]}            \
            --seconds {params[2]}
        """

rule nn_pathways:
    input:
        "{data_dir}/{type}/{size}/{filter}/counts.training.tsv",
        "{data_dir}/{type}/{size}/{filter}/{labels}.training.tsv",
        "{data_dir}/{type}/{size}/{filter}/counts.validation.tsv",
        "{data_dir}/{type}/{size}/{filter}/{labels}.validation.tsv",
    output:
        "{data_dir}/{type}/{size}/{filter}/pw_{architecture}/{labels}/loss.tsv",
    params:
        "{data_dir}/{type}/{size}/{filter}/pw_{architecture}/{labels}",
        config['pathways'],
        config['max_training_time'],
        config['max_epochs'],
        config['batch_size']
    conda:
        "envs/py_data.yaml",
    shell:
        """
        mkdir -p {params[0]}

        python3 scripts/nn_pathways.py                          \
            --train_features        {input[0]}                  \
            --train_labels          {input[1]}                  \
            --validate_features     {input[2]}                  \
            --validate_labels       {input[3]}                  \
            --output_dir            {params[0]}                 \
            --pathways              {params[1]}                 \
            --linear_architecture   {wildcards.architecture}    \
            --max_seconds           {params[2]}                 \
            --max_epochs            {params[3]}                 \
            --batch_size            {params[4]}
        """

rule run_nn:
    input:
        expand("{data_dir}/{type}/{size}/{filter}/nn_{architecture}/loss.tsv",
                data_dir = config['data_dir'],
                type = config['type'],
                size = config['size'],
                filter = config['filter'],
                architecture = config['architecture'],
                ),
