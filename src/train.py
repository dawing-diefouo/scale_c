"""
Hauptskript - orchestriert alle Module für H5P-Generator Training
"""

from src.config import Config
from src.utils import setup_logging, save_config
from src.data_loader import DatasetLoader
from src.model_setup import ModelSetup
from src.preprocessing import DataPreprocessor
from src.trainer import ModelTrainer


def main():
    # 1. Konfiguration
    config = Config()

    # 2. Logging setup
    logger = setup_logging(config.training.output_dir)
    logger.info("=" * 60)
    logger.info("🎓 H5P-Generator Training")
    logger.info("=" * 60)
    logger.info(f"Modell: {config.model.base_model}")
    logger.info(f"Dataset: {config.data.train_path}")
    logger.info(f"Output: {config.training.output_dir}")
    logger.info(f"Max Length: {config.data.max_length}")
    logger.info(f"Batch Size: {config.training.batch_size}")
    logger.info(f"Gradient Accumulation: {config.training.gradient_accumulation_steps}")
    logger.info(f"Effektive Batch Size: {config.training.batch_size * config.training.gradient_accumulation_steps}")
    logger.info(f"Epochen: {config.training.num_epochs}")
    logger.info(f"Learning Rate: {config.training.learning_rate}")
    logger.info("=" * 60)

    try:
        # 3. Config speichern
        save_config(config, config.training.output_dir)
        logger.info("✓ Konfiguration gespeichert")

        # 4. Daten laden
        data_loader = DatasetLoader(config.data, logger)
        train_dataset = data_loader.load_train_data()
        eval_dataset = data_loader.load_eval_data()

        # 5. Modell setup
        model_setup = ModelSetup(config.model, config.lora, logger)
        model, tokenizer = model_setup.setup()

        # 6. Daten preprocessen
        preprocessor = DataPreprocessor(tokenizer, config.data.max_length)

        logger.info("🧹 Tokenisiere Trainingsdaten...")
        train_tokenized = preprocessor.process_dataset(train_dataset)
        logger.info(f"✓ Training tokenisiert: {len(train_tokenized)} Beispiele")

        eval_tokenized = None
        if eval_dataset:
            logger.info("🧹 Tokenisiere Evaluationsdaten...")
            eval_tokenized = preprocessor.process_dataset(eval_dataset)
            logger.info(f"✓ Evaluation tokenisiert: {len(eval_tokenized)} Beispiele")

        # 7. Training
        trainer_instance = ModelTrainer(config.training, logger)
        trainer_instance.train(model, tokenizer, train_tokenized, eval_tokenized)

        # 8. Zusammenfassung
        logger.info("=" * 60)
        logger.info("🎉 Training erfolgreich abgeschlossen!")
        logger.info("=" * 60)
        logger.info(f"📁 Modell: {config.training.output_dir.resolve()}")
        logger.info(f"📊 Trainingsdaten: {len(train_dataset)} Beispiele")
        logger.info(f"🔧 LoRA Rank: {config.lora.r}")
        logger.info(f"📈 Epochen: {config.training.num_epochs}")
        logger.info("=" * 60)
        logger.info("Nächste Schritte:")
        logger.info("1. Teste das Modell: python inference.py")
        logger.info("2. Validiere generierte H5Ps")
        logger.info("3. Bei Bedarf: Nachtraining mit mehr Daten")
        logger.info("=" * 60)

    except Exception as e:
        logger.error(f"❌ Fehler: {e}", exc_info=True)
        raise

if __name__ == "__main__":
    main()