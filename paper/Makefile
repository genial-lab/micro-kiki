NAME = micro-kiki

all: $(NAME).pdf

$(NAME).pdf: $(NAME).tex
	pdflatex -interaction=nonstopmode $(NAME).tex
	bibtex $(NAME) || true
	pdflatex -interaction=nonstopmode $(NAME).tex
	pdflatex -interaction=nonstopmode $(NAME).tex

clean:
	rm -f $(NAME).{aux,bbl,blg,log,out,pdf,toc}

.PHONY: all clean
