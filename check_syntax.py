import ast

try:
    with open(r'd:\pregrad major project\app.py', 'r') as f:
        ast.parse(f.read())
    print('Syntax OK')
except SyntaxError as e:
    print(f'Syntax Error: {e}')
