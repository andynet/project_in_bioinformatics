configfile: "config.yaml"

wildcard_constraints:
    study="TCGA|GTEx",
    type="exon|gene",
    size="full|[0-9]+x[0-9]+",
    filter="pca|naive",
    nn="ff|pw",
    architecture="[0-9-]+",
    labels="samples|stages",

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
        "{data_dir}/{type}/{size}/pca/counts.tsv",
        "{data_dir}/{type}/{size}/naive/counts.tsv",
        "{data_dir}/{type}/{size}/TCGA/dummy/samples.tsv",
        "{data_dir}/{type}/{size}/TCGA/dummy/stages.tsv",
    output:
        expand("{{data_dir}}/{{type}}/{{size}}/pca/counts.{set}.tsv",
                set = ['training', 'validation', 'testing']),
        expand("{{data_dir}}/{{type}}/{{size}}/naive/counts.{set}.tsv",
                set = ['training', 'validation', 'testing']),
        expand("{{data_dir}}/{{type}}/{{size}}/TCGA/dummy/samples.{set}.tsv",
                set = ['training', 'validation', 'testing']),
        expand("{{data_dir}}/{{type}}/{{size}}/TCGA/dummy/stages.{set}.tsv",
                set = ['training', 'validation', 'testing']),
    params:
        "{data_dir}/{type}/{size}/histogram_data"
    conda:
        "envs/py_data.yaml",
    shell:
        """
        mkdir -p {params[0]}

        python3 scripts/split_datasets.py   \
            --naive {input[1]}              \
            --pca {input[0]}                \
            --samples {input[2]}            \
            --stages {input[3]}             \
            --histogram_data {params[0]}
        """

rule nn_feedforward:
    input:
        "{data_dir}/{type}/{size}/{filter}/counts.training.tsv",
        "{data_dir}/{type}/{size}/TCGA/dummy/{labels}.training.tsv",
        "{data_dir}/{type}/{size}/{filter}/counts.validation.tsv",
        "{data_dir}/{type}/{size}/TCGA/dummy/{labels}.validation.tsv",
    output:
        "{data_dir}/{type}/{size}/{filter}/ff_{architecture}/{labels}/loss.tsv",
    params:
        "{data_dir}/{type}/{size}/{filter}/ff_{architecture}/{labels}",
        config['max_training_time'],
        config['max_epochs'],
        config['batch_size'],
    conda:
        "envs/py_data.yaml",
    shell:
        """
        mkdir -p {params[0]}/models

        python3 scripts/nn_feedforward.py                       \
            --features_training     {input[0]}                  \
            --labels_training       {input[1]}                  \
            --features_validation   {input[2]}                  \
            --labels_validation     {input[3]}                  \
            --out_dir               {params[0]}                 \
            --architecture          {wildcards.architecture}    \
            --max_training_time     {params[1]}                 \
            --max_epochs            {params[2]}                 \
            --batch_size            {params[3]}

        """

rule nn_pathways:
    input:
        "{data_dir}/{type}/{size}/naive/counts.training.tsv",
        "{data_dir}/{type}/{size}/TCGA/dummy/{labels}.training.tsv",
        "{data_dir}/{type}/{size}/naive/counts.validation.tsv",
        "{data_dir}/{type}/{size}/TCGA/dummy/{labels}.validation.tsv",
    output:
        "{data_dir}/{type}/{size}/naive/pw_{architecture}/{labels}/loss.tsv",
    params:
        "{data_dir}/{type}/{size}/naive/pw_{architecture}/{labels}",
        config['pathways'],
        config['max_training_time'],
        config['max_epochs'],
        config['batch_size'],
    conda:
        "envs/py_data.yaml",
    shell:
        """
        mkdir -p {params[0]}/models

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
        expand("{data_dir}/{type}/{size}/{analysis_type}/loss.tsv",
                data_dir = config['data_dir'],
                type = config['type'],
                size = config['size'],
                analysis_type = config['analysis_type']
                ),
