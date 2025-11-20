
import torch
from config import TrainingConfig, LearningRateConfig
from learning_rate import get_lr
from gpt import GPT, GPTConfig

def verify_lr():
    print("Verifying learning rate configuration...")
    
    # Initialize configs
    lr_config = LearningRateConfig(
        warmup_steps=10,
        total_steps=100,
        max_lr=1.0,
        min_lr=0.1
    )
    training_config = TrainingConfig()
    training_config.lr_config = lr_config
    
    gpt_config = GPTConfig(vocab_size=100, n_layers=1, n_heads=1, d_model=16)
    
    # Create model
    model = GPT(gpt_config)
    
    # Create optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=1.0, 
        weight_decay=0.01
    )
    
    # Create scheduler
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda step: get_lr(step, lr_config)
    )
    
    # Step through and verify
    print("Stepping through scheduler...")
    
    # Step 0
    current_lr = scheduler.get_last_lr()[0]
    print(f"Step 0 LR: {current_lr:.6f}")
    assert current_lr == 0.0, f"Expected 0.0, got {current_lr}"
    
    # Step 5 (Warmup)
    for _ in range(5):
        optimizer.step()
        scheduler.step()
    
    current_lr = scheduler.get_last_lr()[0]
    print(f"Step 5 LR: {current_lr:.6f}")
    expected_lr = 1.0 * 5 / 10
    assert abs(current_lr - expected_lr) < 1e-6, f"Expected {expected_lr}, got {current_lr}"
    
    # Step 10 (End of warmup)
    for _ in range(5):
        optimizer.step()
        scheduler.step()
        
    current_lr = scheduler.get_last_lr()[0]
    print(f"Step 10 LR: {current_lr:.6f}")
    assert abs(current_lr - 1.0) < 1e-6, f"Expected 1.0, got {current_lr}"
    
    print("✓ Learning rate verification successful!")

if __name__ == "__main__":
    verify_lr()
