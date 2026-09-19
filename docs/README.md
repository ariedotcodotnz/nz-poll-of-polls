# Developer documentation

The NZ Poll of Polls aggregates published opinion polls into an estimate of party support and forecasts
seats and coalitions for the New Zealand general election on 7 November 2026.

These pages are for people running or changing the code, and are read here on GitHub. The public website,
built with Quarto from [`../website/`](../website/), has the forecast and the two pages on method:
[How the model works](https://ariedotcodotnz.github.io/nz-poll-of-polls/model.html) and [How it is tested](https://ariedotcodotnz.github.io/nz-poll-of-polls/evaluation.html).

| Page | Read it to |
|---|---|
| [Getting started](getting-started.md) | install the package and Quarto, run the pipeline and view the website |
| [Pipeline](pipeline.md) | understand each stage, the command line and caching |
| [Configuration](configuration.md) | change elections, pollsters, electorates, priors or variants |
| [Data](data.md) | know where the polls and results come from and how they are cleaned |
| [Outputs](outputs.md) | find a column in any output file |
| [Operations](operations.md) | run it in GitHub Actions, publish it, or prepare the next election |
| [Development](development.md) | work on the code or the website, run the tests or add a model variant |

The original R implementation of the 2023 model is kept, unmaintained, in [`../legacy-r/`](../legacy-r/).
