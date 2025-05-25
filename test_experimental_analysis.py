"""
Test and demonstrate the experimental approaches analysis
"""

import sys
sys.path.append('.')

from src.evolution.experimental_analysis import ExperimentalApproachAnalyzer, ExperimentalApproach

def run_feasibility_analysis():
    print("🔬 Phase 3: Experimental Approaches Feasibility Analysis")
    print("=" * 60)
    
    analyzer = ExperimentalApproachAnalyzer()
    
    # Analyze each approach
    print("\n📊 GRAPH NEURAL NETWORKS ANALYSIS")
    print("-" * 40)
    gnn_analysis = analyzer.analyze_gnn_approach()
    print(f"Overall Score: {gnn_analysis.overall_score:.3f}")
    print(f"Technical Feasibility: {gnn_analysis.technical_feasibility:.3f}")
    print(f"Chess Domain Fit: {gnn_analysis.chess_domain_fit:.3f}")
    print(f"Computational Cost: {gnn_analysis.computational_cost:.3f}")
    print(f"Key Advantages:")
    for adv in gnn_analysis.advantages[:3]:
        print(f"  ✅ {adv}")
    print(f"Key Challenges:")
    for challenge in gnn_analysis.key_challenges[:2]:
        print(f"  ⚠️ {challenge}")
    
    print("\n🧠 MEMORY-AUGMENTED NETWORKS ANALYSIS")
    print("-" * 40)
    memory_analysis = analyzer.analyze_memory_augmented_approach()
    print(f"Overall Score: {memory_analysis.overall_score:.3f}")
    print(f"Expected Performance: {memory_analysis.expected_performance:.3f}")
    print(f"Implementation Complexity: {memory_analysis.implementation_complexity:.3f}")
    print(f"Key Advantages:")
    for adv in memory_analysis.advantages[:3]:
        print(f"  ✅ {adv}")
    print(f"Key Challenges:")
    for challenge in memory_analysis.key_challenges[:2]:
        print(f"  ⚠️ {challenge}")
    
    print("\n🏁 CO-EVOLUTION ANALYSIS")
    print("-" * 40)
    coevo_analysis = analyzer.analyze_coevolution_approach()
    print(f"Overall Score: {coevo_analysis.overall_score:.3f}")
    print(f"Chess Domain Fit: {coevo_analysis.chess_domain_fit:.3f}")
    print(f"Evolutionary Compatibility: {coevo_analysis.evolutionary_compatibility:.3f}")
    print(f"Key Advantages:")
    for adv in coevo_analysis.advantages[:3]:
        print(f"  ✅ {adv}")
    print(f"Key Challenges:")
    for challenge in coevo_analysis.key_challenges[:2]:
        print(f"  ⚠️ {challenge}")
    
    print("\n📈 COMPARATIVE ANALYSIS")
    print("-" * 40)
    comparison = analyzer.generate_comparative_analysis()
    
    print("Overall Rankings:")
    for approach, score in sorted(comparison["overall_rankings"].items(), 
                                 key=lambda x: x[1], reverse=True):
        print(f"  {approach}: {score:.3f}")
    
    print("\nCategory Leaders:")
    for category, rankings in comparison["category_rankings"].items():
        best = max(rankings, key=rankings.get)
        score = rankings[best]
        print(f"  {category}: {best} ({score:.3f})")
    
    print("\n🎯 RECOMMENDATIONS")
    print("-" * 40)
    recs = comparison["recommendations"]
    print(f"Immediate Implementation: {recs['immediate_implementation']}")
    print(f"Research Priority: {recs['research_priority']}")
    print(f"Hybrid Approach: {recs['hybrid_approach']}")
    print(f"Long-term Goal: {recs['long_term_goal']}")
    
    print("\n🗺️ IMPLEMENTATION STRATEGY")
    print("-" * 40)
    strategy = comparison["implementation_strategy"]
    print(f"Phase 1: {strategy['phase_1']}")
    print(f"Phase 2: {strategy['phase_2']}")
    print(f"Phase 3: {strategy['phase_3']}")
    
    print("\n✅ Analysis Complete!")
    return comparison

if __name__ == "__main__":
    run_feasibility_analysis()