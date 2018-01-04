#!/bin/bash

cd ex"$1"/TeX
pdflatex ex"$1".tex
pdflatex ex"$1".tex
rm *.aux *.log *.out
mv ex"$1".pdf ..
cd ../..