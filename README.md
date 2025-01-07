# Description
Generic Python template for AWE Group

## :gear: Installation

### Dependencies

- Sphinx Dependencies, see [requirements](requirements.txt)

## Installation Instructions

1. **Clone the repository**:
    ```bash
    git clone https://github.com/ocayon/EKF-AWE
    ```

2. **Navigate to the repository folder**:
    ```bash
    cd quasi-steady-awes
    ```

3. **Create a virtual environment**:

   - **Linux or Mac**:
     ```bash
     python3 -m venv venv
     ```
   - **Windows**:
     ```bash
     python -m venv venv
     ```

4. **Activate the virtual environment**:

   - **Linux or Mac**:
     ```bash
     source venv/bin/activate
     ```
   - **Windows**:
     ```bash
     .\venv\Scripts\activate
     ```

5. **Install the required dependencies**:

   - For users:
     ```bash
     pip install .
     ```
   - For developers:
     ```bash
     pip install -e .[dev]
     ```

6. **To deactivate the virtual environment**:
    ```bash
    deactivate
    ```

## :eyes: Usage

```python
import foobar

# returns 'words'
foobar.pluralize('word')

# returns 'geese'
foobar.pluralize('goose')

# returns 'phenomenon'
foobar.singularize('phenomena')
```
## :wave: Contributing (optional)

Pull requests are welcome. For major changes, please open an issue first
to discuss what you would like to change.

Please make sure to update tests as appropriate.

## :warning: License and Waiver

Specify the license under which your software is distributed, and include the copyright notice:

> Technische Universiteit Delft hereby disclaims all copyright interest in the program “NAME PROGRAM” (one line description of the content or function) written by the Author(s).
> 
> Prof.dr. H.G.C. (Henri) Werij, Dean of Aerospace Engineering
> 
> Copyright (c) [YEAR] [NAME SURNAME].

## :gem: Help and Documentation
[AWE Group | Developer Guide](https://awegroup.github.io/developer-guide/)


