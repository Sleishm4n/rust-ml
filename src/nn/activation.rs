pub fn relu(x: f32) -> f32 {
    if x > 0.0 {
        x
    } else {
        0.0
    }
}

pub fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

pub fn tanh(x: f32) -> f32 {
    (x.exp() - (-x).exp()) / (x.exp() + (-x).exp())
}

pub fn d_relu(x: f32) -> f32 {
    if x > 0.0 { 1.0 } else { 0.0 }
}

pub fn d_sigmoid(x: f32) -> f32 {
    sigmoid(x) * (1.0 - sigmoid(x))
}

pub fn d_tanh(x: f32) -> f32 {
    1.0 - tanh(x).powi(2) 
}
