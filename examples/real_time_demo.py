import asyncio
import httpx
import sys
import json
import itertools

API_BASE = "http://127.0.0.1:8000"

# ANSI Colors
CYAN = "\033[96m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
MAGENTA = "\033[95m"
RESET = "\033[0m"
BOLD = "\033[1m"

spinner = itertools.cycle(["-", "\\", "|", "/"])

TEST_CASES = [
    {
        "name": f"{MAGENTA}The Deep Synthesis Test{RESET}",
        "query": "What are the core architectural differences between FastAPI and Litestar? Which one is faster? Provide technical reasons."
    },
    {
        "name": f"{CYAN}The Multi-Tool Reasoning Test{RESET}",
        "query": "Calculate the compound interest of $15,000 at 7% over 10 years. Then, research the current US inflation rate and calculate the estimated real return."
    },
    {
        "name": f"{YELLOW}Strict Isolation (Job A){RESET}",
        "query": "Write down the phrase 'THE EAGLE HAS LANDED' into your memory tool."
    },
    {
        "name": f"{YELLOW}Strict Isolation (Job B){RESET}",
        "query": "Write down the phrase 'THE CONDOR HAS FLOWN' into your memory tool."
    }
]

async def dispatch_job(client: httpx.AsyncClient, query: str) -> str:
    response = await client.post(f"{API_BASE}/research", json={"question": query})
    response.raise_for_status()
    data = response.json()
    return data["job_id"]

async def poll_job(client: httpx.AsyncClient, job_id: str, name: str):
    print(f"{BOLD}Started {name} {RESET}(ID: {job_id})")
    while True:
        response = await client.get(f"{API_BASE}/status/{job_id}")
        response.raise_for_status()
        data = response.json()
        
        status = data["status"]
        if status == "done":
            print(f"\n{GREEN}[OK] {name} COMPLETED!{RESET}")
            report = data["result"]
            print(f"  {BOLD}Topic:{RESET} {report['topic']}")
            print(f"  {BOLD}Summary:{RESET} {report['summary']}")
            for finding in report.get("findings", []):
                print(f"    - {finding}")
            print("-" * 60)
            break
        elif status == "failed":
            print(f"\n{YELLOW}[FAILED] {name} FAILED!{RESET}")
            print(data)
            break
            
        sys.stdout.write(f"\r  {next(spinner)} {name} is {status}...   ")
        sys.stdout.flush()
        await asyncio.sleep(0.5)

async def main():
    print(f"\n{BOLD}{CYAN}=== CEREBRO API REAL-TIME DEMO ==={RESET}\n")
    
    # Check if server is running
    try:
        async with httpx.AsyncClient() as client:
            await client.get(f"{API_BASE}/health")
    except Exception:
        print(f"{YELLOW}Error: Could not connect to Cerebro API at {API_BASE}.{RESET}")
        print("Please start the server using: uvicorn api.app:app --host 0.0.0.0 --port 8000")
        sys.exit(1)
        
    print(f"Dispatching {len(TEST_CASES)} jobs concurrently...\n")
    
    async with httpx.AsyncClient(timeout=60.0) as client:
        # Dispatch all jobs at the same time
        job_ids = []
        for case in TEST_CASES:
            job_id = await dispatch_job(client, case["query"])
            job_ids.append((job_id, case["name"]))
            
        # Poll them all concurrently
        tasks = [poll_job(client, j_id, name) for j_id, name in job_ids]
        await asyncio.gather(*tasks)
        
    print(f"{BOLD}{GREEN}All real-time tests completed successfully!{RESET}\n")

if __name__ == "__main__":
    asyncio.run(main())
