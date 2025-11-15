
from InteractiveTrainer.display import TrainerDisplay
from InteractiveTrainer.interactive_trainer import InteractiveTrainer

def main():
    """Entry point"""
    display = TrainerDisplay()
    mode = display.get_training_mode()
    use_lstm = display.get_architecture()
    display.show_config(mode, use_lstm)
    
    trainer = InteractiveTrainer(
        use_triplet_loss=(mode == 'triplet'),
        use_lstm=use_lstm
    )
    trainer.interactive_loop()


if __name__ == "__main__":
    main()