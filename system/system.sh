for n in {1..12}
do
html2text -o system_"$n".txt system_"$n".html
done
exit 0
