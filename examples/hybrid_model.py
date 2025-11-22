import sys
import os
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../'))
from pycore.tikzeng import *

# Define custom colors for quantum and special components
def to_custom_colors():
    return r"""
\def\ConvColor{rgb:yellow,5;red,2.5;white,5}
\def\ConvReluColor{rgb:yellow,5;red,5;white,5}
\def\PoolColor{rgb:red,1;black,0.3}
\def\FcColor{rgb:blue,5;red,2.5;white,5}
\def\FcReluColor{rgb:blue,5;red,5;white,4}
\def\SoftmaxColor{rgb:magenta,5;black,7}
\def\QuantumColor{rgb:cyan,5;magenta,3;white,2}
\def\FusionColor{rgb:orange,5;yellow,3;white,2}
\def\NormColor{rgb:green,3;blue,2;white,5}
\def\OutputColor{rgb:green,5;white,3}
\def\ClassicalColor{rgb:blue,3;white,7}
"""

# Custom Quantum Layer
def to_Quantum(name, s_filer=256, n_filer=2, offset="(0,0,0)", to="(0,0,0)", width=3, height=25, depth=25, caption="Quantum"):
    return r"""
\pic[shift={""" + offset + r"""}] at """ + to + r""" 
    {Box={
        name=""" + name + r""",
        caption=""" + caption + r""",
        xlabel={{""" + str(n_filer) + r""", }},
        zlabel=""" + str(s_filer) + r""",
        fill=\QuantumColor,
        height=""" + str(height) + r""",
        width=""" + str(width) + r""",
        depth=""" + str(depth) + r"""
        }
    };
"""

# Custom Fusion Layer
def to_Fusion(name, offset="(0,0,0)", to="(0,0,0)", radius=3.5, opacity=0.8):
    return r"""
\pic[shift={""" + offset + r"""}] at """ + to + r""" 
    {Ball={
        name=""" + name + r""",
        fill=\FusionColor,
        opacity=""" + str(opacity) + r""",
        radius=""" + str(radius) + r""",
        logo=$\alpha$
        }
    };
"""

# Custom Layer Norm
def to_LayerNorm(name, n_filer=2, offset="(0,0,0)", to="(0,0,0)", width=1.5, height=20, depth=20, caption="LayerNorm"):
    return r"""
\pic[shift={""" + offset + r"""}] at """ + to + r""" 
    {Box={
        name=""" + name + r""",
        caption=""" + caption + r""",
        xlabel={{""" + str(n_filer) + r"""}},
        fill=\NormColor,
        height=""" + str(height) + r""",
        width=""" + str(width) + r""",
        depth=""" + str(depth) + r"""
        }
    };
"""

# Architecture for Hybrid Model with dual paths
arch = [
    to_head('..'),
    to_custom_colors(),
    to_begin(),
    
    # ============ INPUT LAYER ============
    to_Conv("input", s_filer=100, n_filer=3, offset="(0,0,0)", to="(0,0,0)", 
            height=50, depth=50, width=1, caption="INPUT"),
    
    # ============ CNN FEATURE EXTRACTOR ============
    # Conv Block 1
    to_Conv("conv1", s_filer=100, n_filer=16, offset="(2,0,0)", to="(input-east)", 
            height=48, depth=48, width=2, caption="Conv1d(3-16)"),
    to_connection("input", "conv1"),
    
    to_Pool("pool1", offset="(0,0,0)", to="(conv1-east)", 
            width=1, height=40, depth=40, opacity=0.5, caption="MaxPool"),
    
    # Conv Block 2
    to_Conv("conv2", s_filer=50, n_filer=32, offset="(1.5,0,0)", to="(pool1-east)", 
            height=35, depth=35, width=3, caption="Conv1d(16-32)"),
    to_connection("pool1", "conv2"),
    
    to_Pool("pool2", offset="(0,0,0)", to="(conv2-east)", 
            width=1, height=28, depth=28, opacity=0.5, caption="MaxPool"),
    
    # Flatten
    to_Conv("flatten", s_filer=736, n_filer=1, offset="(1,0,0)", to="(pool2-east)", 
            height=30, depth=8, width=1, caption="Flatten"),
    to_connection("pool2", "flatten"),
    
    # ============ DUAL OUTPUT FROM CNN ============
    # Quantum Parameters Branch
    to_Conv("fc_quantum", s_filer=2, n_filer=1, offset="(2,3,0)", to="(flatten-east)", 
            height=20, depth=20, width=2, caption="FC(736-2) Quantum"),
    to_connection("flatten", "fc_quantum"),
    
    # Feature Map Selector Branch
    to_Conv("selector", s_filer=3, n_filer=1, offset="(2,-3,0)", to="(flatten-east)", 
            height=18, depth=18, width=2.5, caption="Selector Net"),
    to_connection("flatten", "selector"),
    
    # ============ PHASE 1 PATH (Classical Only) ============
    to_Conv("classical_path", s_filer=2, n_filer=1, offset="(3,6,0)", to="(fc_quantum-east)", 
            height=18, depth=18, width=2, caption="PHASE 1 Classical"),
    
    # ============ PHASE 2 PATH (Quantum) ============
    # Quantum Layer
    to_Quantum("quantum_layer", s_filer=23, n_filer=2, offset="(3,0,0)", to="(fc_quantum-east)", 
               width=4, height=25, depth=25, caption="Quantum Layer"),
    to_connection("fc_quantum", "quantum_layer"),
    to_connection("selector", "quantum_layer"),
    
    # Classical Features for Fusion
    to_Conv("classical_features", s_filer=2, n_filer=1, offset="(0,3,0)", to="(quantum_layer-north)", 
            height=15, depth=15, width=1.5, caption="Classical Mean"),
    
    # ============ FUSION MECHANISM ============
    to_Fusion("fusion", offset="(3,0,0)", to="(quantum_layer-east)", radius=3.5, opacity=0.85),
    to_connection("quantum_layer", "fusion"),
    to_connection("classical_features", "fusion"),
    
    # Add quantum weight annotation
    to_Conv("qw_param", s_filer=1, n_filer=1, offset="(0,-3.5,0)", to="(fusion-south)", 
            height=12, depth=12, width=1, caption="alpha weight"),
    
    # ============ LAYER NORMALIZATION ============
    to_LayerNorm("layer_norm", n_filer=2, offset="(3,0,0)", to="(fusion-east)", 
                 width=2, height=22, depth=22, caption="LayerNorm"),
    to_connection("fusion", "layer_norm"),
    
    # Connect Phase 1 path to layer norm
    to_connection("classical_path", "layer_norm"),
    
    # ============ FINAL CLASSIFIER ============
    to_Conv("classifier", s_filer=2, n_filer=2, offset="(2.5,0,0)", to="(layer_norm-east)", 
            height=20, depth=20, width=2.5, caption="Classifier FC(2-2)"),
    to_connection("layer_norm", "classifier"),
    
    # ============ OUTPUT ============
    to_SoftMax("output", s_filer=2, offset="(2,0,0)", to="(classifier-east)", 
               width=2, height=5, depth=20, opacity=0.9, 
               caption="Output"),
    to_connection("classifier", "output"),
    
    # ============ ANNOTATIONS ============
    # Add text annotations for phases
    r"""
% Phase 1 annotation
\node[text width=3.5cm, align=center, fill=blue!10, rounded corners] at (6, 8, 0) 
    {\textbf{PHASE 1} \\ Classical \\ 30 epochs};

% Phase 2 annotation  
\node[text width=3.5cm, align=center, fill=cyan!10, rounded corners] at (12, 8, 0) 
    {\textbf{PHASE 2} \\ Quantum \\ 50 epochs};

% Warmup annotation
\node[text width=3.5cm, align=center, fill=orange!10, rounded corners] at (12, -6, 0) 
    {\textbf{WARMUP} \\ First 10 epochs};

% Parameter count
\node[text width=4cm, align=center, fill=green!10, rounded corners] at (18, 8, 0) 
    {\textbf{Parameters: 50,794} \\ CNN: 99.93\% \\ Quantum: 0.045\%};

% Fusion formula
\node[text width=4cm, align=center, fill=orange!10, rounded corners] at (12, -8, 0) 
    {\textbf{Fusion} \\ $(1-\alpha) \times classical$ \\ $+ \alpha \times quantum$};
""",
    
    to_end()
]

def main():
    namefile = str(sys.argv[0]).split('.')[0]
    to_generate(arch, namefile + '.tex')
    print(f"\n{'='*60}")
    print(f"Generated LaTeX file: {namefile}.tex")
    print(f"{'='*60}")
    print("\nTo compile the diagram:")
    print(f"1. Run: bash ../tikzmake.sh {namefile}")
    print(f"2. Or manually: pdflatex {namefile}.tex")
    print("\nArchitecture Overview:")
    print("- Input: (batch, 3, 100)")
    print("- CNN Feature Extractor: 50,757 parameters")
    print("- Quantum Layer: 23 parameters")
    print("- Dual Training Phases:")
    print("  * Phase 1: Classical training (30 epochs)")
    print("  * Phase 2: Quantum fine-tuning (50 epochs)")
    print("- Output: Binary classification (Coding vs Intergenic)")
    print(f"{'='*60}\n")

if __name__ == '__main__':
    main()