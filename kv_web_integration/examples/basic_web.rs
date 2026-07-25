use kv_web_core::{KvWeb, TokenId, EdgeKind};
use kv_web_runtime::KvWebRuntime;
use kv_web_integration::{KvCache, KvWebIntegration};

fn main() {
    // ---------------------------------------------------------
    // 1. Build a fake KV-cache (keys + values)
    // ---------------------------------------------------------
    let cache = KvCache {
        keys: vec![
            vec![0.1, 0.2], // token 0
            vec![0.3, 0.4], // token 1
            vec![0.5, 0.6], // token 2
            vec![0.7, 0.8], // token 3
            vec![0.9, 1.0], // token 4
        ],
        values: vec![
            vec![1.1, 1.2],
            vec![1.3, 1.4],
            vec![1.5, 1.6],
            vec![1.7, 1.8],
            vec![1.9, 2.0],
        ],
    };

    // ---------------------------------------------------------
    // 2. Build the KV Web
    // ---------------------------------------------------------
    let mut web = KvWeb::new();

    // Node A: tokens 0,1
    let node_a = web.create_node(
        vec![TokenId(0), TokenId(1)],
        Some("intro"),
        0.9,
    );

    // Node B: tokens 2,3
    let node_b = web.create_node(
        vec![TokenId(2), TokenId(3)],
        Some("topic"),
        0.7,
    );

    // Node C: token 4
    let node_c = web.create_node(
        vec![TokenId(4)],
        Some("detail"),
        0.5,
    );

    // ---------------------------------------------------------
    // 3. Add edges (webbing)
    // ---------------------------------------------------------
    web.add_edge(node_a, node_b, 1.0, EdgeKind::Semantic);
    web.add_edge(node_b, node_c, 0.8, EdgeKind::Semantic);

    // ---------------------------------------------------------
    // 4. Use the integration layer
    // ---------------------------------------------------------
    let integration = KvWebIntegration::new(&web, &cache);

    // Region: start at node A, depth 2
    let region_tokens = web.tokens_in_region(node_a, 2);
    println!("Tokens in region (depth 2): {:?}", region_tokens);

    // Hard attention mask
    let hard_mask = integration.attention_mask(node_a, 2);
    println!("Hard attention mask: {:?}", hard_mask);

    // Soft attention mask
    let soft_mask = integration.soft_attention_mask(node_a, 2);
    println!("Soft attention mask: {:?}", soft_mask);

    // KV subset
    let (keys_subset, values_subset) = integration.kv_subset(node_a, 2);
    println!("KV subset (keys): {:?}", keys_subset);
    println!("KV subset (values): {:?}", values_subset);

    // ---------------------------------------------------------
    // 5. Region score
    // ---------------------------------------------------------
    let score = web.region_score(node_a, 2);
    println!("Region score: {}", score);
}
