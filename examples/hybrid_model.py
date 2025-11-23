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

# Architecture for Hybrid Model matching HybridQuantumGenomicModel
arch = [
    to_head('..'),
    to_custom_colors(),
    to_begin(),
    
    # ============ INPUT LAYER ============
    to_Conv("input", s_filer=100, n_filer=3, offset="(0,0,0)", to="(0,0,0)", 
            height=50, depth=50, width=1.5, caption="Input\\\\3×100"),
    
    # ============ CNN FEATURE EXTRACTOR (GenomicCNN) ============
    # Conv Block 1: 3→64
    to_Conv("conv1", s_filer=100, n_filer=64, offset="(2,0,0)", to="(input-east)", 
            height=48, depth=48, width=2.5, caption="Conv1d\\\\3→64\\\\k=5"),
    to_connection("input", "conv1"),
    
    to_Pool("pool1", offset="(1.2,0,0)", to="(conv1-east)", 
            width=1, height=42, depth=42, opacity=0.5, caption="MaxPool\\\\k=2"),
    to_connection("conv1", "pool1"),
    
    # Conv Block 2: 64→128
    to_Conv("conv2", s_filer=50, n_filer=128, offset="(2,0,0)", to="(pool1-east)", 
            height=38, depth=38, width=3, caption="Conv1d\\\\64→128\\\\k=5"),
    to_connection("pool1", "conv2"),
    
    to_Pool("pool2", offset="(1.2,0,0)", to="(conv2-east)", 
            width=1, height=32, depth=32, opacity=0.5, caption="MaxPool\\\\k=2"),
    to_connection("conv2", "pool2"),
    
    # Conv Block 3: 128→256
    to_Conv("conv3", s_filer=25, n_filer=256, offset="(2,0,0)", to="(pool2-east)", 
            height=28, depth=28, width=3.5, caption="Conv1d\\\\128→256\\\\k=3"),
    to_connection("pool2", "conv3"),
    
    to_Pool("pool3", offset="(1.2,0,0)", to="(conv3-east)", 
            width=1, height=22, depth=22, opacity=0.5, caption="MaxPool\\\\k=2"),
    to_connection("conv3", "pool3"),
    
    # Flatten
    to_Conv("flatten", s_filer=800, n_filer=1, offset="(1.5,0,0)", to="(pool3-east)", 
            height=25, depth=10, width=1, caption="Flatten\\\\~800"),
    to_connection("pool3", "flatten"),
    
    # ============ BRIDGE LAYER ============
    to_Conv("bridge", s_filer=512, n_filer=1, offset="(2,0,0)", to="(flatten-east)", 
            height=30, depth=30, width=2, caption="{\\small FC}\\\\800→512"),
    to_connection("flatten", "bridge"),
    
    # ============ DUAL PATH SPLIT ============
    
    # === QUANTUM PATH (HybridFeatureMapQuantumCircuit) ===
    # Preprocessing
    to_Conv("fc_preprocess", s_filer=256, n_filer=1, offset="(2.5,3,0)", to="(bridge-east)", 
            height=28, depth=28, width=2, caption="{\\small FC}\\\\512→256"),
    to_connection("bridge", "fc_preprocess"),
    
    # Feature Map Selector (learnable weights)
    to_Conv("selector", s_filer=3, n_filer=1, offset="(2.5,-3,0)", to="(bridge-east)", 
            height=15, depth=15, width=2, caption="Selector\\\\Softmax(3)"),
    to_connection("bridge", "selector"),
    
    # Quantum Layer (3 feature maps: Z, ZZ, Pauli)
    to_Quantum("quantum_layer", s_filer=12, n_filer=2, offset="(3,0,0)", to="(fc_preprocess-east)", 
               width=5, height=32, depth=32, caption="Quantum\\\\3 Maps"),
    to_connection("fc_preprocess", "quantum_layer"),
    to_connection("selector", "quantum_layer"),
    
    # Post-quantum processing
    to_Conv("fc_post", s_filer=256, n_filer=1, offset="(3,0,0)", to="(quantum_layer-east)", 
            height=28, depth=28, width=2, caption="{\\small FC}\\\\12→256"),
    to_connection("quantum_layer", "fc_post"),
    
    # Quantum output
    to_Conv("fc_out_quantum", s_filer=2, n_filer=1, offset="(2,0,0)", to="(fc_post-east)", 
            height=20, depth=20, width=2, caption="{\\small FC}\\\\256→2"),
    to_connection("fc_post", "fc_out_quantum"),
    
    # === CLASSICAL RESIDUAL PATH ===
    to_Conv("residual", s_filer=2, n_filer=1, offset="(2.5,8,0)", to="(bridge-east)", 
            height=20, depth=20, width=2, caption="Residual\\\\512→2"),
    to_connection("bridge", "residual"),
    
    # ============ FUSION WITH QUANTUM WEIGHT ============
    to_Fusion("fusion", offset="(2.5,0,0)", to="(fc_out_quantum-east)", radius=3.5, opacity=0.85),
    to_connection("fc_out_quantum", "fusion"),
    to_connection("residual", "fusion"),
    
    # Quantum weight annotation (the α parameter)
    r"""
\node[text width=5cm, align=center, yshift=-3cm] at (fusion-south) 
    {\small $\alpha \times$ quantum + $(1-\alpha) \times$ classical};
""",
    
    # ============ OUTPUT ============
    to_SoftMax("output", s_filer=2, offset="(3,0,0)", to="(fusion-east)", 
               width=2, height=8, depth=20, opacity=0.9, 
               caption="Output\\\\2 classes"),
    to_connection("fusion", "output"),
    
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