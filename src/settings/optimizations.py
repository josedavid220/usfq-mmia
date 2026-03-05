import os


def is_remote_vscode():
    """Detect if running in VSCode remote container/SSH environment.

    Uses DMI Product information to distinguish between VSCode Remote and Google Colab.
    Google Colab runs on "Google Compute Engine" VMs.
    """
    try:
        if os.path.exists("/sys/class/dmi/id/product_name"):
            with open("/sys/class/dmi/id/product_name", "r") as f:
                product = f.read().strip()
                print("DMI:", product)
                # Google Colab runs on Google Compute Engine
                if "Google" in product or "Compute Engine" in product:
                    return False  # This is Google Colab
        # If not Google Compute Engine, assume VSCode Remote/Local
        return True
    except:
        # If we can't read DMI, assume local/non-Colab
        return True


# Detect environment and set optimal configurations
IS_REMOTE = is_remote_vscode()
print(f"Environment detected: {'VSCode Remote' if IS_REMOTE else 'Google Colab'}")

# Performance-optimized settings based on environment
if IS_REMOTE:
    NUM_WORKERS = 2  #  to avoid IPC overhead
    PIN_MEMORY = False  # Reduce memory pressure in container
    PERSISTENT_WORKERS = False
    PREFETCH_FACTOR = None
    ENABLE_PROGRESS_BAR = False  # Disable progress bar updates over network
    LOG_EVERY_N_STEPS = 50  # Reduce logging frequency
    CALLBACK_VERBOSE = True
else:
    NUM_WORKERS = 4  # Leverage multiple cores for data loading
    PIN_MEMORY = True  # Faster GPU transfer
    PERSISTENT_WORKERS = True  # Reuse worker processes
    PREFETCH_FACTOR = 2  # Load batches ahead
    ENABLE_PROGRESS_BAR = True
    LOG_EVERY_N_STEPS = 10
    CALLBACK_VERBOSE = True

print(f"DataLoader num_workers set to: {NUM_WORKERS}")
print("Progress bar set to", ENABLE_PROGRESS_BAR)
