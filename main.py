from interactive.interactive_trainer import InteractiveTrainer
def main():

    trainer = InteractiveTrainer()
    trainer.interactive_loop()


if __name__ == "__main__":
    main()


## TODO  birleşik sözcükleri hallet
## TODO  kişi çekimlemeleri gerekli mi?
## TODO  iken ekini hallet
## TODO iyor ile gelen düzensizlikleri hallet
## TODO ek hiyerarsisini gözden geçir. evdekiler  evdekinden gibi ekler alabilir ki mesela.
## -dık  sıfat fiil
## boş dönen ekleri hallet, 2.tekil emir kipi, n ve ünlü ile biten giili ıg eki gibi. s