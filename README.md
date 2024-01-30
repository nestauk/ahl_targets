# 💙 Targeting the health of the nation

## 📋 About this Project

In this project we explored options for how health targets could be effectively implemented and incentivised within the food sector. This work will included examining the current policy landscape, including existing voluntary target programmes and health commitments already made by industry. This will help inform the development of our potential target options, recognising the need to balance impact with feasibility, for both business and government. We modelled the impact of these target options and estimate their effect on daily calories purchased and future population obesity levels. This highlighted which targets would have the greatest impact on reducing obesity. All project outputs are published on the [Nesta website](https://www.nesta.org.uk/project/industry-targets-to-improve-health/).

## :hammer_and_wrench: Setup

**Step 1.** Install the following components:

- [conda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/download.html)
- [direnv](https://formulae.brew.sh/formula/direnv#default), for using environment variables

**Step 2.** Run the following command from the repo root folder:

```
make install
```

This will configure the development environment:

- Setup the conda environment with the name `ahl_targets`
- Configure pre-commit actions (for example, running a code formatter before each commit)
- Configure metaflow

**Step 3.** Activate the newly created conda environment and you're good to go!

```shell
$ conda activate ahl_targets
```

## :handshake: Contributor guidelines

[Technical and working style guidelines](https://github.com/nestauk/ds-cookiecutter/blob/master/GUIDELINES.md)

---

<small><p>Project based on <a target="_blank" href="https://github.com/nestauk/ds-cookiecutter">Nesta's data science project template</a>
(<a href="http://nestauk.github.io/ds-cookiecutter">Read the docs here</a>).
</small>
