# Where to Put run.sh: Best Practices

## TL;DR: Put it in the project folder ✅

```
✅ RECOMMENDED:
problems/
  └── YOUR_PROJECT/
      ├── run.sh                    ← Put it here!
      ├── input/
      │   ├── evaluate.py
      │   └── src/
      │       └── init_program.py
      └── configs/
          └── config.yaml

❌ NOT RECOMMENDED:
science-codeevolve/
  ├── run.sh                        ← Don't put it here
  └── problems/
      └── YOUR_PROJECT/
          └── ...
```

## Why Project Folder is Better

### ✅ Advantages

1. **Self-Contained Projects**
   - Everything for one project is in one place
   - No confusion about which project you're running
   
2. **Easy Sharing**
   - Share just `problems/YOUR_PROJECT/` folder
   - Colleague can drop it in and run immediately
   - No need to share entire repository
   
3. **Parallel Execution**
   ```bash
   # Run multiple projects at once
   cd problems/project_A && bash run.sh &
   cd problems/project_B && bash run.sh &
   cd problems/project_C && bash run.sh &
   ```
   
4. **Project-Specific Settings**
   - Each project can have different:
     - CPU affinity settings
     - Output directories
     - Checkpoint policies
   - No need to edit global settings
   
5. **Simple Workflow**
   ```bash
   cd problems/YOUR_PROJECT
   bash run.sh
   ```
   vs
   ```bash
   # Edit PROJECT_NAME in root run.sh every time
   nano run.sh
   bash run.sh
   ```

6. **Version Control**
   - Project-specific configs tracked with project
   - Easy to see what changed per project
   - Better git history

### ❌ Root Folder Problems

1. **One Project at a Time**
   - Can only run one project
   - Must edit PROJECT_NAME each time
   
2. **Not Portable**
   - Can't share just one project
   - Need entire repo structure
   
3. **Confusion**
   - Which PROJECT_NAME is set?
   - Did I remember to change it?
   
4. **Conflicts**
   - Multiple people can't run different projects
   - Git conflicts on single run.sh file

## How to Set Up

### Step 1: Copy Template to Project

```bash
cp problems/run_template.sh problems/YOUR_PROJECT/run.sh
```

### Step 2: Edit Project Name

```bash
cd problems/YOUR_PROJECT
nano run.sh
```

Change this line:
```bash
PROJECT_NAME="YOUR_PROJECT"  # e.g., "F_time"
```

### Step 3: Run

```bash
# From project folder
cd problems/YOUR_PROJECT
bash run.sh

# Or from anywhere
bash problems/YOUR_PROJECT/run.sh
```

## Alternative: Template at Root (For Reference Only)

You can keep a template at root for reference, but **copy it to projects before use**:

```
science-codeevolve/
  ├── run_template.sh              ← Template (don't run directly)
  └── problems/
      ├── F_time/
      │   └── run.sh               ← Copy template here, customize & run
      ├── project_A/
      │   └── run.sh               ← Copy template here, customize & run
      └── project_B/
          └── run.sh               ← Copy template here, customize & run
```

## Real-World Example

### Team Scenario

**Alice** working on F_time:
```bash
cd problems/F_time
bash run.sh  # Runs F_time with its settings
```

**Bob** working on optimization problem:
```bash
cd problems/optimization
bash run.sh  # Runs optimization with its settings
```

**Both run simultaneously, no conflicts!**

### Single User, Multiple Experiments

```bash
# Terminal 1: Run baseline
cd problems/my_problem
bash run.sh  # Uses config.yaml

# Terminal 2: Run with meta-prompting
cd problems/my_problem
# Edit run.sh to use config_mp.yaml
bash run.sh  # Different config, same project

# Both run at the same time!
```

## Summary

| Aspect | Project Folder | Root Folder |
|--------|---------------|-------------|
| Portability | ✅ Share just project | ❌ Need whole repo |
| Parallel runs | ✅ Multiple at once | ❌ One at a time |
| Clarity | ✅ Always clear | ❌ Edit each time |
| Team work | ✅ No conflicts | ❌ File conflicts |
| Simplicity | ✅ `cd` and run | ❌ Edit then run |

**Recommendation: Always put run.sh in the project folder.**

---

See `problems/README.md` for complete documentation.
