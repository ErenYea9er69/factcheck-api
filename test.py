import asyncio
import json
from main import check_claims, CheckRequest

async def run_test():
    print("Running verification directly from backend...")
    req = CheckRequest(text="The moon landing happened in 1969. Also, humans have successfully established a permanent colony on Mars in 2024. Python was invented by Guido van Rossum.")
    
    # This directly calls the backend logic
    results = await check_claims(req)
    
    # Convert pydantic models to dict
    results_dict = [r.model_dump() for r in results]
    
    with open("output.json", "w") as f:
        json.dump(results_dict, f, indent=4)
        
    print("Success! Results saved to output.json")

if __name__ == "__main__":
    asyncio.run(run_test())
