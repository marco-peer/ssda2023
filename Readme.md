## ReadMe

Competition of  SSDA 2023 in Fribourg, Switzerland. - HTR using TrOCR and the API provided by HuggingFace. Results are included in the .tsv file. The dataset of the challenge was the [BullingerDB](https://tc11.cvc.uab.es/datasets/BullingerDB_1), a challenging historical dataset for handwritten text recognition. 

### execute

Training of TrOCR can be started via 

~~~
  python main.py
~~~

and the transcriptions on the test set (as a .tsv) are obtained by running

~~~
  python eval.py
~~~

Refer to the files for checking the settings.

### Resources

Li, M., Lv, T., Chen, J., Cui, L., Lu, Y., Florencio, D., Zhang, C., Li, Z., & Wei, F. (2023). **TrOCR: Transformer-Based Optical Character Recognition with Pre-trained Models.** Proceedings of the AAAI Conference on Artificial Intelligence, 37(11), 13094-13102. https://doi.org/10.1609/aaai.v37i11.26538
