# Model Training

## Overview

This document details the LLM fine-tuning process using Ollama with LoRA (Low-Rank Adaptation) for our movie recommendation system. We'll fine-tune a base language model to understand user preferences and generate personalized movie recommendations.

## Model Architecture

### Base Model Selection
- **Primary**: Llama 2 7B or Llama 3 8B (via Ollama)
- **Alternative**: Mistral 7B for faster inference
- **Reasoning**: Balance between performance and computational efficiency

### LoRA Configuration
- **Rank (r)**: 16-64 (configurable based on task complexity)
- **Alpha**: 32 (scaling factor for LoRA weights)
- **Target Modules**: Query, Key, Value, and Output projections
- **Dropout**: 0.1 for regularization

## Training Data Format

### Instruction-Response Pairs
```python
def format_training_example(user_history: List[Dict], target_movie: Dict) -> Dict[str, str]:
    """Format training data for instruction tuning."""
    
    # Extract user preferences
    liked_movies = [m['title'] for m in user_history if m['rating'] >= 4.0]
    disliked_movies = [m['title'] for m in user_history if m['rating'] <= 2.0]
    
    # Genre analysis
    genre_preferences = analyze_genre_preferences(user_history)
    
    instruction = f"""You are a movie recommendation expert. Based on a user's viewing history, predict if they would like a specific movie and provide a rating from 1-5.

User's Profile:
- Liked movies: {', '.join(liked_movies[:5])}
- Disliked movies: {', '.join(disliked_movies[:3])}
- Preferred genres: {', '.join(genre_preferences['liked'][:3])}
- Avoided genres: {', '.join(genre_preferences['disliked'][:2])}

Movie to evaluate: {target_movie['title']} ({target_movie['year']})
Genres: {target_movie['genres']}
Plot: {target_movie.get('plot', 'Not available')}

Provide your prediction:"""

    response = f"""Rating Prediction: {target_movie['actual_rating']}/5

Reasoning: Based on the user's preference for {genre_preferences['liked'][0]} movies and their positive ratings for similar films like {liked_movies[0]}, this movie aligns well with their taste. The {target_movie['genres']} genre combination matches their viewing patterns, and the movie's themes are consistent with their preferences.

Confidence: {calculate_confidence_score(user_history, target_movie)}/10"""

    return {
        "instruction": instruction,
        "response": response,
        "user_id": target_movie['user_id'],
        "movie_id": target_movie['movie_id']
    }
```

### Dataset Preparation
```python
import json
from typing import List, Dict, Tuple
import pandas as pd

class TrainingDataGenerator:
    def __init__(self, ratings_df: pd.DataFrame, movies_df: pd.DataFrame):
        self.ratings = ratings_df
        self.movies = movies_df
        
    def generate_training_examples(self, min_history: int = 10, 
                                 max_examples_per_user: int = 50) -> List[Dict]:
        """Generate training examples for LoRA fine-tuning."""
        
        training_examples = []
        
        for user_id in self.ratings['userId'].unique():
            user_ratings = self.ratings[self.ratings['userId'] == user_id].sort_values('timestamp')
            
            if len(user_ratings) < min_history:
                continue
                
            # Create sliding window examples
            for i in range(min_history, min(len(user_ratings), min_history + max_examples_per_user)):
                history = user_ratings.iloc[:i]
                target = user_ratings.iloc[i]
                
                # Enrich with movie metadata
                history_with_meta = self.enrich_with_metadata(history)
                target_with_meta = self.enrich_with_metadata(pd.DataFrame([target]))
                
                example = format_training_example(
                    history_with_meta.to_dict('records'),
                    target_with_meta.iloc[0].to_dict()
                )
                
                training_examples.append(example)
        
        return training_examples
    
    def enrich_with_metadata(self, ratings_subset: pd.DataFrame) -> pd.DataFrame:
        """Add movie metadata to ratings."""
        return ratings_subset.merge(self.movies, on='movieId', how='left')
```

## LoRA Fine-tuning Implementation

### Configuration Setup
```python
from peft import LoraConfig, get_peft_model, TaskType
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments
import torch

class LoRATrainer:
    def __init__(self, model_name: str = "llama2:7b", device: str = "cuda"):
        self.model_name = model_name
        self.device = device
        self.setup_model()
        
    def setup_model(self):
        """Initialize model and tokenizer with LoRA configuration."""
        
        # Load base model (assuming Ollama model is available locally)
        self.tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")
        self.model = AutoModelForCausalLM.from_pretrained(
            "meta-llama/Llama-2-7b-hf",
            torch_dtype=torch.float16,
            device_map="auto"
        )
        
        # LoRA configuration
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=32,  # Rank
            lora_alpha=64,  # Scaling factor
            lora_dropout=0.1,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            bias="none"
        )
        
        # Apply LoRA to model
        self.model = get_peft_model(self.model, lora_config)
        
        # Add padding token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
    
    def prepare_dataset(self, training_examples: List[Dict]) -> torch.utils.data.Dataset:
        """Prepare dataset for training."""
        
        class RecommendationDataset(torch.utils.data.Dataset):
            def __init__(self, examples, tokenizer, max_length=2048):
                self.examples = examples
                self.tokenizer = tokenizer
                self.max_length = max_length
            
            def __len__(self):
                return len(self.examples)
            
            def __getitem__(self, idx):
                example = self.examples[idx]
                
                # Format as conversation
                text = f"### Instruction:\n{example['instruction']}\n\n### Response:\n{example['response']}"
                
                # Tokenize
                encoding = self.tokenizer(
                    text,
                    truncation=True,
                    padding="max_length",
                    max_length=self.max_length,
                    return_tensors="pt"
                )
                
                return {
                    "input_ids": encoding["input_ids"].flatten(),
                    "attention_mask": encoding["attention_mask"].flatten(),
                    "labels": encoding["input_ids"].flatten()
                }
        
        return RecommendationDataset(training_examples, self.tokenizer)
```

### Training Configuration
```python
def setup_training_arguments() -> TrainingArguments:
    """Configure training parameters for LoRA fine-tuning."""
    
    return TrainingArguments(
        output_dir="./models/lora-movie-recommender",
        num_train_epochs=3,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        gradient_accumulation_steps=4,
        warmup_steps=100,
        learning_rate=2e-4,
        fp16=True,
        logging_steps=10,
        evaluation_strategy="steps",
        eval_steps=500,
        save_steps=1000,
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        report_to="wandb",  # For Weights & Biases integration
        run_name="lora-movie-recommender-v1"
    )

def train_lora_model(trainer_instance: LoRATrainer, 
                    training_examples: List[Dict],
                    validation_examples: List[Dict]):
    """Execute LoRA fine-tuning process."""
    
    from transformers import Trainer, DataCollatorForLanguageModeling
    
    # Prepare datasets
    train_dataset = trainer_instance.prepare_dataset(training_examples)
    eval_dataset = trainer_instance.prepare_dataset(validation_examples)
    
    # Data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=trainer_instance.tokenizer,
        mlm=False
    )
    
    # Initialize trainer
    trainer = Trainer(
        model=trainer_instance.model,
        args=setup_training_arguments(),
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        tokenizer=trainer_instance.tokenizer
    )
    
    # Start training
    trainer.train()
    
    # Save the fine-tuned model
    trainer.save_model()
    trainer_instance.tokenizer.save_pretrained("./models/lora-movie-recommender")
    
    return trainer
```

## Ollama Integration

### Model Deployment to Ollama
```python
import subprocess
import json
from pathlib import Path

class OllamaModelManager:
    def __init__(self, model_path: str):
        self.model_path = Path(model_path)
        
    def create_modelfile(self, base_model: str = "llama2:7b") -> str:
        """Create Ollama Modelfile for custom model."""
        
        modelfile_content = f"""FROM {base_model}

# Set custom parameters
PARAMETER temperature 0.7
PARAMETER top_p 0.9
PARAMETER top_k 40
PARAMETER num_ctx 4096

# System prompt for movie recommendations
SYSTEM \"\"\"You are an expert movie recommendation assistant. You analyze user preferences and provide personalized movie recommendations with detailed explanations. Always provide ratings on a 1-5 scale and explain your reasoning based on the user's viewing history and preferences.\"\"\"

# Load LoRA adapter
ADAPTER {self.model_path / "adapter_model.bin"}
"""
        
        modelfile_path = self.model_path / "Modelfile"
        with open(modelfile_path, 'w') as f:
            f.write(modelfile_content)
            
        return str(modelfile_path)
    
    def deploy_to_ollama(self, model_name: str = "movie-recommender:latest"):
        """Deploy fine-tuned model to Ollama."""
        
        modelfile_path = self.create_modelfile()
        
        # Create model in Ollama
        cmd = f"ollama create {model_name} -f {modelfile_path}"
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        
        if result.returncode == 0:
            print(f"Successfully deployed model: {model_name}")
            return model_name
        else:
            raise Exception(f"Failed to deploy model: {result.stderr}")
    
    def test_model(self, model_name: str, test_prompt: str) -> str:
        """Test the deployed model with a sample prompt."""
        
        cmd = f'ollama run {model_name} "{test_prompt}"'
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        
        return result.stdout if result.returncode == 0 else result.stderr
```

## Model Evaluation

### Evaluation Metrics
```python
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error
from scipy.stats import pearsonr, spearmanr

class ModelEvaluator:
    def __init__(self, model_name: str):
        self.model_name = model_name
        
    def evaluate_predictions(self, test_examples: List[Dict]) -> Dict[str, float]:
        """Evaluate model performance on test set."""
        
        predictions = []
        actual_ratings = []
        
        for example in test_examples:
            # Get model prediction
            predicted_rating = self.get_model_prediction(example['instruction'])
            predictions.append(predicted_rating)
            actual_ratings.append(example['actual_rating'])
        
        # Calculate metrics
        metrics = {
            'rmse': np.sqrt(mean_squared_error(actual_ratings, predictions)),
            'mae': mean_absolute_error(actual_ratings, predictions),
            'pearson_correlation': pearsonr(actual_ratings, predictions)[0],
            'spearman_correlation': spearmanr(actual_ratings, predictions)[0]
        }
        
        return metrics
    
    def get_model_prediction(self, prompt: str) -> float:
        """Extract rating prediction from model response."""
        
        cmd = f'ollama run {self.model_name} "{prompt}"'
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        
        # Parse rating from response
        response = result.stdout
        rating_match = re.search(r'Rating.*?(\d+(?:\.\d+)?)', response)
        
        if rating_match:
            return float(rating_match.group(1))
        else:
            return 3.0  # Default neutral rating
```

## Training Pipeline

### Complete Training Workflow
```python
def run_training_pipeline(data_path: str, output_path: str):
    """Execute complete model training pipeline."""
    
    # 1. Load and prepare data
    ratings = pd.read_csv(f"{data_path}/processed_ratings.csv")
    movies = pd.read_csv(f"{data_path}/processed_movies.csv")
    
    # 2. Generate training examples
    data_generator = TrainingDataGenerator(ratings, movies)
    training_examples = data_generator.generate_training_examples()
    
    # 3. Split data
    train_size = int(0.8 * len(training_examples))
    val_size = int(0.1 * len(training_examples))
    
    train_examples = training_examples[:train_size]
    val_examples = training_examples[train_size:train_size + val_size]
    test_examples = training_examples[train_size + val_size:]
    
    # 4. Initialize trainer and train model
    trainer = LoRATrainer()
    trained_model = train_lora_model(trainer, train_examples, val_examples)
    
    # 5. Deploy to Ollama
    model_manager = OllamaModelManager(output_path)
    model_name = model_manager.deploy_to_ollama()
    
    # 6. Evaluate model
    evaluator = ModelEvaluator(model_name)
    metrics = evaluator.evaluate_predictions(test_examples)
    
    # 7. Log results
    print(f"Training completed. Model: {model_name}")
    print(f"Evaluation metrics: {metrics}")
    
    return model_name, metrics
```

## Hyperparameter Optimization

### Automated Tuning
```python
import optuna
from typing import Dict, Any

def objective(trial: optuna.Trial) -> float:
    """Objective function for hyperparameter optimization."""
    
    # Suggest hyperparameters
    lora_rank = trial.suggest_int('lora_rank', 8, 64, step=8)
    lora_alpha = trial.suggest_int('lora_alpha', 16, 128, step=16)
    learning_rate = trial.suggest_float('learning_rate', 1e-5, 5e-4, log=True)
    batch_size = trial.suggest_categorical('batch_size', [2, 4, 8])
    
    # Train model with suggested parameters
    trainer = LoRATrainer()
    # ... configure with trial parameters
    
    # Return validation loss
    return validation_loss

def optimize_hyperparameters(n_trials: int = 20):
    """Run hyperparameter optimization."""
    
    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=n_trials)
    
    return study.best_params
```

## Next Steps

1. **Advanced Techniques**: Implement QLoRA for even more efficient training
2. **Multi-task Learning**: Add additional tasks like genre classification
3. **Continual Learning**: Implement online learning for model updates
4. **Model Compression**: Apply quantization for faster inference
5. **A/B Testing**: Set up framework for comparing model versions

This training pipeline provides a robust foundation for fine-tuning LLMs for movie recommendations using LoRA and Ollama.
