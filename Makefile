all: report.pdf


report.pdf: report.tex report.bib
	pdflatex report.tex 
	bibtex report.aux
