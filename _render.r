# from Jupyter notebook to R markdown
rmarkdown:::convert_ipynb(
    input = "python/01-prob_regression.ipynb", 
    output = "Rmarkdown/01-prob_regression.rmd"
)

# pdf (slides)
rmarkdown::render(
    input = "Rmarkdown/01-prob_regression.rmd",
    output_file = "prob_regression.pdf",
    output_dir = "slide",
    clean = TRUE,
    encoding = "utf8"
)
