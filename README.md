# 🇹🇷 TÜRK KELİME ANALİZİ (Turkish Word Analysis)

*Bu README dosyası, projenin Türkçe sürümünü takiben aşağıda İngilizce sürümünü içermektedir.*

## Genel Bakış

**turkishWords**, Türkçe kelimelerin morfolojik analizine ve ayrıştırılmasına odaklanmış bir Doğal Dil İşleme (DDİ) projesidir.

Türkçe, kelimelerin bir kök kelimeye birden fazla ek (sonek) eklenerek oluşturulduğu eklemeli bir dildir. Bu depo, karmaşık Türkçe kelimeleri bir makine öğrenimi yaklaşımı kullanarak köklerine ve eklenmiş morfemlerine ayırmak için gerekli mantığı, modeli ve eğitim betiklerini sağlar.

## Temel Özellikler

- **Morfolojik Ayrıştırma**: Bir Türkçe kelimeyi köküne ve ek zincirine ayırmak için temel mantık. (`decomposition.py` ve `suffixes.py` dosyalarına bakınız).
- **Makine Öğrenimi Sıralama Modeli**: Birden fazla aday analiz olduğunda en olası morfolojik analizleri sıralamak için eğitilmiş bir model (`turkish_morph_model.pt`) kullanarak doğruluğu artırır. (`ml_ranking_model.py` dosyasına bakınız).
- **Etkileşimli Eğitim Arayüzü**: Yüksek kaliteli eğitim verileri oluşturmak için ayrıştırma sonuçlarını manuel olarak inceleme ve düzeltme aracı. (`interactive_trainer.py` dosyasına bakınız).
- **Kelime Dağarcığı**: Analiz ve eğitim amaçlı bir Türkçe kelime dağarcığı (`words.txt`) içerir.

## Kurulum

Bu proje Python ile yazılmıştır ve PyTorch dahil olmak üzere standart bilimsel kütüphaneler gerektirir.

### Ön Koşullar

Python 3.8+ sürümünün yüklü olması gerekir.

### Kurulum Adımları

1. Depoyu klonlayın:
```bash
git clone https://github.com/oktayelb/turkishWords.git
cd turkishWords
```

2. Bağımlılıkları yükleyin:

Morfolojik model ve eğitim betiklerini çalıştırmak için aşağıdaki paketler gereklidir:
```bash
pip install torch numpy tqdm
```

## Kullanım

### 1. Bir Türkçe Kelimeyi Analiz Etme

Temel ayrıştırma ve sıralama özelliklerini kullanmak için genellikle `decomposition.py` ve `ml_ranking_model.py` dosyalarındaki mantığı entegre etmeniz gerekir.

```bash
# Örnek ayrıştırma betiği çalıştırma (lütfen gerçek kullanımla değiştirin)
# python decomposition.py --word "okuldakilerle"
```

### 2. Etkileşimli Eğitim

Modeli geliştirmek veya yeni etiketlenmiş veriler oluşturmak isterseniz, etkileşimli eğiticiyi kullanın:

```bash
python interactive_trainer.py
```

Bu betik, muhtemelen modelin ayrıştırma önerilerini incelemek ve insan düzeltmeleri sağlamak için rehberli bir arayüz sunar.

## Proje Yapısı

| Dosya/Klasör | Açıklama |
|--------------|----------|
| `decomposition.py` | Kelime ayrıştırma adayları için temel mantığı yönetir. |
| `suffixes.py` | Türkçe eklere ait tanımları ve kuralları içerir. |
| `ml_ranking_model.py` | Ayrıştırma olasılıklarını sıralamak için kullanılan Makine Öğrenimi kodu. |
| `turkish_morph_model.pt` | Morfolojik sıralama için önceden eğitilmiş PyTorch modeli. |
| `interactive_trainer.py` | Etkileşimli veri etiketleme ve model geliştirme betiği. |
| `suffix_vocab.json` | Bilinen eklerin kelime dağarcığını içeren JSON dosyası. |
| `words.txt` | Test ve eğitim için kullanılan Türkçe kelime dağarcığı (korpus). |
| `old_versions/` | Kodun önceki yinelemelerini depolamak için dizin. |
| `speech2text/` | Potansiyel konuşmadan metne entegrasyon veya deneyler için dizin. |

## Katkıda Bulunma

Katkılarınız memnuniyetle karşılanır! Önerileriniz, hata raporlarınız varsa veya kod katkısında bulunmak istiyorsanız, lütfen bir sorun (issue) açmaktan veya bir çekme isteği (pull request) göndermekten çekinmeyin.


---

# 🇬🇧 ENGLISH VERSION

*This README contains the English version of the project details, following the Turkish version above.*

## Overview

**turkishWords** is a Natural Language Processing (NLP) project focused on the morphological analysis and decomposition of Turkish words.

Turkish is an agglutinative language, meaning words are formed by adding multiple suffixes to a root word. This repository provides the logic, model, and training scripts necessary to decompose complex Turkish words into their root and affixed morphemes using a machine learning approach.

## Key Features

- **Morphological Decomposition**: Core logic for splitting a Turkish word into its root and suffix chain. (See `decomposition.py` and `suffixes.py`).
- **Machine Learning Ranking Model**: Utilizes a trained model (`turkish_morph_model.pt`) to rank the most plausible morphological analyses when multiple candidates exist, improving accuracy. (See `ml_ranking_model.py`).
- **Interactive Training Interface**: A tool for manually reviewing and correcting decomposition results to generate high-quality training data. (See `interactive_trainer.py`).
- **Word Corpus**: Includes a corpus of Turkish words (`words.txt`) for analysis and training purposes.

## Installation

This project is written in Python and requires standard scientific libraries, including PyTorch.

### Prerequisites

You should have Python 3.8+ installed.

### Setup

1. Clone the repository:
```bash
git clone https://github.com/oktayelb/turkishWords.git
cd turkishWords
```

2. Install dependencies:

The following packages are required to run the morphological model and training scripts:
```bash
pip install torch numpy tqdm
```

## Usage

### 1. Analyzing a Turkish Word

To use the core decomposition and ranking features, you would typically integrate the logic from `decomposition.py` and `ml_ranking_model.py`.

```bash
# Example of running a decomposition script (replace with actual usage)
# python decomposition.py --word "okuldakilerle"
```

### 2. Interactive Training

If you wish to improve the model or generate new labeled data, use the interactive trainer:

```bash
python interactive_trainer.py
```

This script likely provides a guided interface for reviewing the model's decomposition suggestions and providing human corrections.

## Project Structure

| File/Folder | Description |
|-------------|-------------|
| `decomposition.py` | Handles the core logic for word decomposition candidates. |
| `suffixes.py` | Contains definitions and rules related to Turkish suffixes. |
| `ml_ranking_model.py` | The Machine Learning code used to rank decomposition possibilities. |
| `turkish_morph_model.pt` | The pre-trained PyTorch model for morphological ranking. |
| `interactive_trainer.py` | Script for interactive data annotation and model improvement. |
| `suffix_vocab.json` | JSON file containing the vocabulary of known suffixes. |
| `words.txt` | A corpus of Turkish words used for testing and training. |
| `old_versions/` | Directory for storing previous iterations of the code. |
| `speech2text/` | Directory for potential speech-to-text integration or experiments. |

## Contributing

Contributions are welcome! If you have suggestions, bug reports, or want to contribute code, please feel free to open an issue or submit a pull request.
