"""
CNN Feature Extractor Architecture Diagram Generator
Uses PlotNeuralNet library to visualize the EnhancedCNN architecture
for genomic sequence classification with quantum feature parameters.
"""

import sys
sys.path.append('../')
from pycore.tikzeng import *

# Custom color definitions for genomic/quantum CNN
def to_custom_colors():
    return r"""
\def\ConvColor{rgb:cyan,5;blue,3;white,5}
\def\ConvReluColor{rgb:cyan,5;blue,5;white,3}
\def\PoolColor{rgb:red,1;black,0.3}
\def\FcColor{rgb:blue,5;cyan,3;white,5}
\def\QuantumColor{rgb:purple,5;blue,3;white,3}
\def\SelectorColor{rgb:orange,5;yellow,3;white,3}
\def\DropoutColor{rgb:gray,5;black,2}
\def\BNColor{rgb:green,3;cyan,2;white,5}
"""

# Custom BatchNorm layer
def to_BatchNorm(name, s_filer=256, n_filer=64, offset="(0,0,0)", to="(0,0,0)", 
                 width=0.5, height=40, depth=40, caption="BN"):
    return r"""
\pic[shift={"""+ offset +"""}] at """+ to +""" 
    {Box={
        name=""" + name +""",
        caption="""+ caption +r""",
        xlabel={{"""+ str(n_filer) +"""}},
        zlabel="""+ str(s_filer) +""",
        fill=\BNColor,
        height="""+ str(height) +""",
        width="""+ str(width) +""",
        depth="""+ str(depth) +""",
        opacity=0.7
        }
    };
"""

# Custom Dropout layer (represented as a thin transparent layer)
def to_Dropout(name, s_filer=256, n_filer=64, offset="(0,0,0)", to="(0,0,0)", 
               width=0.3, height=40, depth=40, caption="Drop"):
    return r"""
\pic[shift={"""+ offset +"""}] at """+ to +""" 
    {Box={
        name=""" + name +""",
        caption="""+ caption +r""",
        xlabel={{"""+ str(n_filer) +"""}},
        zlabel="""+ str(s_filer) +""",
        fill=\DropoutColor,
        height="""+ str(height) +""",
        width="""+ str(width) +""",
        depth="""+ str(depth) +""",
        opacity=0.4
        }
    };
"""

# Custom Flatten representation
def to_Flatten(name, offset="(0,0,0)", to="(0,0,0)", width=1, height=15, depth=15, 
               n_features=736, caption="Flatten"):
    return r"""
\pic[shift={"""+ offset +"""}] at """+ to +""" 
    {Box={
        name=""" + name +""",
        caption="""+ caption +r""",
        xlabel={{"""+ str(n_features) +"""}},
        fill={rgb:gray,3;white,7},
        height="""+ str(height) +""",
        width="""+ str(width) +""",
        depth="""+ str(depth) +""",
        opacity=0.6
        }
    };
"""

# Custom Fully Connected layer with special coloring
def to_FC_Quantum(name, n_input=736, n_output=2, offset="(0,0,0)", to="(0,0,0)", 
                  width=2, height=8, depth=8, caption="FC"):
    return r"""
\pic[shift={"""+ offset +"""}] at """+ to +""" 
    {Box={
        name=""" + name +""",
        caption="""+ caption +r""",
        xlabel={{"""+ str(n_output) +"""}},
        zlabel="""+ str(n_input) +""",
        fill=\QuantumColor,
        height="""+ str(height) +""",
        width="""+ str(width) +""",
        depth="""+ str(depth) +""",
        opacity=0.8
        }
    };
"""

# Custom Selector Network representation
def to_FC_Selector(name, n_input=736, n_output=64, offset="(0,0,0)", to="(0,0,0)", 
                   width=2, height=10, depth=10, caption="FC"):
    return r"""
\pic[shift={"""+ offset +"""}] at """+ to +""" 
    {Box={
        name=""" + name +""",
        caption="""+ caption +r""",
        xlabel={{"""+ str(n_output) +"""}},
        zlabel="""+ str(n_input) +""",
        fill=\SelectorColor,
        height="""+ str(height) +""",
        width="""+ str(width) +""",
        depth="""+ str(depth) +""",
        opacity=0.8
        }
    };
"""

# Output node representation
def to_Output(name, n_output=2, offset="(0,0,0)", to="(0,0,0)", 
              width=1.5, height=6, depth=6, caption="Output", color="\QuantumColor"):
    return r"""
\pic[shift={"""+ offset +"""}] at """+ to +""" 
    {Box={
        name=""" + name +""",
        caption="""+ caption +r""",
        xlabel={{"""+ str(n_output) +"""}},
        fill=""" + color +""",
        height="""+ str(height) +""",
        width="""+ str(width) +""",
        depth="""+ str(depth) +""",
        opacity=0.9
        }
    };
"""

# Text annotation
def to_annotation(text, position="(0,10,0)"):
    return r"""
\node[text width=8cm, align=center] at """ + position + r""" {""" + text + r"""};
"""

# Main architecture definition
arch = [
    to_head('..'),
    to_custom_colors(),
    to_begin(),
    
    # Title annotation
    to_annotation(r"\Large \textbf{CNN Feature Extractor (EnhancedCNN)}", "(0,12,0)"),
    to_annotation(r"\small Genomic Sequence to Quantum Parameters", "(0,11,0)"),
    
    # Input layer (3 channels, sequence length 100)
    to_Conv("input", s_filer=100, n_filer=3, offset="(0,0,0)", to="(0,0,0)", 
            height=50, depth=50, width=1.5, caption="Input\\\\3×100"),
    
    # Annotation for input channels
    to_annotation(r"\tiny Ch0: Base\\Ch1: Pu/Py\\Ch2: H-Bond", "(-2,0,0)"),
    
    # ===== FIRST CONVOLUTIONAL BLOCK =====
    # Conv1D: 3 -> 16 channels, kernel=5
    to_Conv("conv1", s_filer=96, n_filer=16, offset="(2,0,0)", to="(input-east)", 
            height=48, depth=48, width=2.5, caption="Conv1D\\\\k=5"),
    to_connection("input", "conv1"),
    
    # Parameters annotation
    to_annotation(r"\tiny 256 params", "(2,5,0)"),
    
    # BatchNorm1D
    to_BatchNorm("bn1", s_filer=96, n_filer=16, offset="(0.3,0,0)", to="(conv1-east)", 
                 width=0.5, height=48, depth=48, caption="BN"),
    to_connection("conv1", "bn1"),
    
    # ReLU (represented as slightly different color band)
    to_Conv("relu1", s_filer=96, n_filer=16, offset="(0.3,0,0)", to="(bn1-east)", 
            height=48, depth=48, width=0.3, caption="ReLU"),
    to_connection("bn1", "relu1"),
    
    # MaxPool (96 -> 48)
    to_Pool("pool1", offset="(0.5,0,0)", to="(relu1-east)", 
            width=1.5, height=42, depth=42, opacity=0.6, caption="MaxPool\\\\k=2"),
    to_connection("relu1", "pool1"),
    
    # Dropout (p=0.3)
    to_Dropout("drop1", s_filer=48, n_filer=16, offset="(0.3,0,0)", to="(pool1-east)", 
               width=0.3, height=42, depth=42, caption="Drop\\\\0.3"),
    to_connection("pool1", "drop1"),
    
    # Output shape annotation
    to_annotation(r"\tiny (16, 48)", "(7.5,-4,0)"),
    
    # ===== SECOND CONVOLUTIONAL BLOCK =====
    # Conv1D: 16 -> 32 channels, kernel=3
    to_Conv("conv2", s_filer=46, n_filer=32, offset="(2,0,0)", to="(drop1-east)", 
            height=38, depth=38, width=3, caption="Conv1D\\\\k=3"),
    to_connection("drop1", "conv2"),
    
    # Parameters annotation
    to_annotation(r"\tiny 1,568 params", "(10,5,0)"),
    
    # BatchNorm1D
    to_BatchNorm("bn2", s_filer=46, n_filer=32, offset="(0.3,0,0)", to="(conv2-east)", 
                 width=0.5, height=38, depth=38, caption="BN"),
    to_connection("conv2", "bn2"),
    
    # ReLU
    to_Conv("relu2", s_filer=46, n_filer=32, offset="(0.3,0,0)", to="(bn2-east)", 
            height=38, depth=38, width=0.3, caption="ReLU"),
    to_connection("bn2", "relu2"),
    
    # MaxPool (46 -> 23)
    to_Pool("pool2", offset="(0.5,0,0)", to="(relu2-east)", 
            width=1.5, height=30, depth=30, opacity=0.6, caption="MaxPool\\\\k=2"),
    to_connection("relu2", "pool2"),
    
    # Dropout (p=0.3)
    to_Dropout("drop2", s_filer=23, n_filer=32, offset="(0.3,0,0)", to="(pool2-east)", 
               width=0.3, height=30, depth=30, caption="Drop\\\\0.3"),
    to_connection("pool2", "drop2"),
    
    # Output shape annotation
    to_annotation(r"\tiny (32, 23)", "(15.5,-4,0)"),
    
    # ===== FLATTEN LAYER =====
    to_Flatten("flatten", offset="(2,0,0)", to="(drop2-east)", 
                width=1.5, height=20, depth=20, n_features=736, caption="Flatten\\\\736"),
    to_connection("drop2", "flatten"),
    
    # ===== DUAL OUTPUT BRANCHES =====
    
    # --- Branch 1: Quantum Parameters ---
    to_FC_Quantum("fc_quantum", n_input=736, n_output=2, offset="(3,3,0)", to="(flatten-east)", 
                  width=2.5, height=12, depth=12, caption="FC\\\\736→2"),
    to_connection("flatten", "fc_quantum"),
    
    # Tanh * π activation
    to_Output("quantum_out", n_output=2, offset="(1.5,0,0)", to="(fc_quantum-east)", 
              width=2, height=10, depth=10, caption="Tanh×π\\\\Quantum", color="\QuantumColor"),
    to_connection("fc_quantum", "quantum_out"),
    
    # Annotation for quantum output
    to_annotation(r"\tiny Range: [-π, π]\\Parameters for\\quantum circuit", "(24,4,0)"),
    to_annotation(r"\tiny 1,474 params", "(20.5,6,0)"),
    
    # --- Branch 2: Feature Map Selector ---
    to_FC_Selector("fc_sel1", n_input=736, n_output=64, offset="(3,-3,0)", to="(flatten-east)", 
                   width=2, height=14, depth=14, caption="FC\\\\736→64"),
    to_connection("flatten", "fc_sel1"),
    
    # Annotation
    to_annotation(r"\tiny 47,168 params", "(20.5,-6,0)"),
    
    # ReLU
    to_Conv("relu_sel", s_filer=64, n_filer=1, offset="(0.5,0,0)", to="(fc_sel1-east)", 
            height=14, depth=14, width=0.3, caption="ReLU"),
    to_connection("fc_sel1", "relu_sel"),
    
    # Dropout (p=0.2)
    to_Dropout("drop_sel", s_filer=64, n_filer=1, offset="(0.3,0,0)", to="(relu_sel-east)", 
               width=0.3, height=14, depth=14, caption="Drop\\\\0.2"),
    to_connection("relu_sel", "drop_sel"),
    
    # FC: 64 -> 3
    to_FC_Selector("fc_sel2", n_input=64, n_output=3, offset="(1,0,0)", to="(drop_sel-east)", 
                   width=2, height=10, depth=10, caption="FC\\\\64→3"),
    to_connection("drop_sel", "fc_sel2"),
    
    # Annotation
    to_annotation(r"\tiny 195 params", "(22.5,-2.5,0)"),
    
    # Softmax output
    to_Output("selector_out", n_output=3, offset="(1.5,0,0)", to="(fc_sel2-east)", 
              width=2, height=9, depth=9, caption="Softmax\\\\Selector", color="\SelectorColor"),
    to_connection("fc_sel2", "selector_out"),
    
    # Annotation for selector output
    to_annotation(r"\tiny Probabilities for:\\Z, ZZ, Pauli\\feature maps", "(26,-3,0)"),
    
    # ===== SUMMARY ANNOTATIONS =====
    to_annotation(r"\textbf{Total Parameters: 50,757}", "(12,-7,0)"),
    to_annotation(r"\tiny Receptive Field: 28 bases", "(12,-8,0)"),
    
    to_end()
]

def main():
    """Generate the LaTeX file for the CNN architecture diagram"""
    namefile = str(sys.argv[0]).split('.')[0]
    to_generate(arch, namefile + '.tex')
    print(f"\n{'='*60}")
    print(f"CNN Feature Extractor diagram generated: {namefile}.tex")
    print(f"{'='*60}")
    print("\nTo compile:")
    print(f"  pdflatex {namefile}.tex")
    print("\nOr use the provided bash script:")
    print(f"  bash ../tikzmake.sh {namefile}")
    print(f"\n{'='*60}")
    print("\nArchitecture Summary:")
    print("  Input:  (batch, 3, 100)")
    print("  Conv1:  (batch, 16, 96)  - 256 params")
    print("  Pool1:  (batch, 16, 48)")
    print("  Conv2:  (batch, 32, 46)  - 1,568 params")
    print("  Pool2:  (batch, 32, 23)")
    print("  Flatten: (batch, 736)")
    print("  Output1: (batch, 2) - Quantum Parameters")
    print("  Output2: (batch, 3) - Feature Map Probabilities")
    print(f"{'='*60}\n")

if __name__ == '__main__':
    main()