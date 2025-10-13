# CV Sport Geometry

**CV Sport Geometry** is a computer vision project that detects key points and geometric structures of **tennis courts** using **NumPy**, **classical computer vision algorithms and techniques**, and **mathematical concepts** such as **linear algebra** - no neural network applied.

The goal of this project is to accurately identify and extract the main lines, intersections, and reference points on a tennis court from images. Details and assumptions in the next chapter.

Key libraries used:
<ul> 
    <li>numpy</li>
    <li>opencv-python</li>
    <li>matplotlib</li>
    <li>pydantic</li>
    <li>scikit-image</li>
    <li>label-studio</li>
</ul>


## Getting Started

Follow the steps below to set up and run the project locally.


### 1. Clone the Repository

Make sure you have **git** installed.  
Then clone the repository and navigate to the project directory:

```bash
git clone https://github.com/username/cv-sport-geometry.git
cd cv-sport-geometry
```

### 2. Requirements
You need to have <a href="https://docs.astral.sh/uv/getting-started/installation/">uv installed</a> on your system to manage the virtual environment and dependencies.
Once it’s installed, you can proceed with the setup.

### 3. Create and Activate the Virtual Environment
```bash
uv venv
source .venv/bin/activate  # Linux / macOS

.venv\Scripts\activate     # Windows
```

### 4. Sync Dependencies
Install all required dependencies using:
```bash
uv sync
```
This will install all packages defined in **pyproject.toml**.

### 5. Run the Project
Depending on how your entry point is defined, you can run the main script with:

```bash
uv run python main.py
```

### Project Structure