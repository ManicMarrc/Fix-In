use std::sync::Mutex;

use macroquad::prelude::*;
use macroquad::rand::ChooseRandom;
use ndarray::prelude::*;
use ndarray_stats::QuantileExt;
use once_cell::sync::Lazy;

fn winc() -> Conf {
  Conf { window_title: "Fix-In".to_string(), window_resizable: false, ..Default::default() }
}

static SPEED_UP: Lazy<Mutex<f32>> = Lazy::new(|| Mutex::new(1.0));

fn get_dt() -> f32 { get_frame_time() * *SPEED_UP.lock().unwrap() }

const PADDLE_SIZE: Vec2 = vec2(96.0, 32.0);
const PADDLE_SPEED: f32 = 350.0;

#[derive(Clone)]
struct Paddle {
  rect: Rect,
}

impl Paddle {
  pub fn new(pos: Vec2) -> Paddle {
    Paddle { rect: Rect::new(pos.x, pos.y, PADDLE_SIZE.x, PADDLE_SIZE.y) }
  }

  pub fn move_x(&mut self, x: f32) {
    self.rect.x = (self.rect.x + x * PADDLE_SPEED * get_dt())
      .clamp(0.0, screen_width().max(PADDLE_SIZE.x * 2.0) - PADDLE_SIZE.x);
  }

  pub fn draw(&self) { draw_rectangle(self.rect.x, self.rect.y, self.rect.w, self.rect.h, BLUE); }
}

const BALL_RADIUS: f32 = 16.0;
const BALL_SPEED: f32 = 200.0;

#[derive(Clone)]
struct Ball {
  circle: Circle,
  velocity: Vec2,
}

impl Ball {
  pub fn new(pos: Vec2, velocity: Vec2) -> Ball {
    Ball { circle: Circle::new(pos.x, pos.y, BALL_RADIUS), velocity }
  }

  pub fn move_x(&mut self) { self.circle.x += self.velocity.x * BALL_SPEED * get_dt(); }

  pub fn move_y(&mut self) { self.circle.y += self.velocity.y * BALL_SPEED * get_dt(); }

  pub fn flip_x(&mut self) {
    self.velocity.x *= -1.0;
    self.move_x();
  }

  pub fn flip_y(&mut self) {
    self.velocity.y *= -1.0;
    self.move_y();
  }

  pub fn draw(&self, color: Color) {
    draw_circle(self.circle.x, self.circle.y, self.circle.r, color);
  }
}

const INPUT_NEURONS: usize = 3;
const HIDDEN_NEURONS_1: usize = 64;
const HIDDEN_NEURONS_2: usize = 64;
const HIDDEN_NEURONS_3: usize = 64;
const HIDDEN_NEURONS_4: usize = 64;
const OUTPUT_NEURONS: usize = 3;

#[derive(Clone)]
struct Mind {
  hidden_weights_1: Array2<f32>,
  hidden_biases_1: Array2<f32>,

  hidden_weights_2: Array2<f32>,
  hidden_biases_2: Array2<f32>,

  hidden_weights_3: Array2<f32>,
  hidden_biases_3: Array2<f32>,

  hidden_weights_4: Array2<f32>,
  hidden_biases_4: Array2<f32>,

  output_weights: Array2<f32>,
  output_biases: Array2<f32>,

  a1: f32,
  a2: f32,
  a3: f32,
  a4: f32,
}

impl Mind {
  pub fn new() -> Mind {
    Mind {
      hidden_weights_1: Array2::from_shape_simple_fn((HIDDEN_NEURONS_1, INPUT_NEURONS), || {
        rand::gen_range(-1.0, 1.0)
      }),
      hidden_biases_1: Array2::from_shape_simple_fn((HIDDEN_NEURONS_1, 1), || {
        rand::gen_range(-1.0, 1.0)
      }),

      hidden_weights_2: Array2::from_shape_simple_fn((HIDDEN_NEURONS_2, HIDDEN_NEURONS_1), || {
        rand::gen_range(-1.0, 1.0)
      }),
      hidden_biases_2: Array2::from_shape_simple_fn((HIDDEN_NEURONS_2, 1), || {
        rand::gen_range(-1.0, 1.0)
      }),

      hidden_weights_3: Array2::from_shape_simple_fn((HIDDEN_NEURONS_3, HIDDEN_NEURONS_2), || {
        rand::gen_range(-1.0, 1.0)
      }),
      hidden_biases_3: Array2::from_shape_simple_fn((HIDDEN_NEURONS_3, 1), || {
        rand::gen_range(-1.0, 1.0)
      }),

      hidden_weights_4: Array2::from_shape_simple_fn((HIDDEN_NEURONS_4, HIDDEN_NEURONS_3), || {
        rand::gen_range(-1.0, 1.0)
      }),
      hidden_biases_4: Array2::from_shape_simple_fn((HIDDEN_NEURONS_4, 1), || {
        rand::gen_range(-1.0, 1.0)
      }),

      output_weights: Array2::from_shape_simple_fn((OUTPUT_NEURONS, HIDDEN_NEURONS_4), || {
        rand::gen_range(-1.0, 1.0)
      }),
      output_biases: Array2::from_shape_simple_fn((OUTPUT_NEURONS, 1), || {
        rand::gen_range(-1.0, 1.0)
      }),

      a1: rand::gen_range(-5.0, 5.0),
      a2: rand::gen_range(-5.0, 5.0),
      a3: rand::gen_range(-5.0, 5.0),
      a4: rand::gen_range(-5.0, 5.0),
    }
  }

  fn breed(&mut self, other: &Mind) {
    for ((i, j), weight) in
      self.hidden_weights_1.indexed_iter_mut().filter(|_| rand::gen_range(0, 1) != 0)
    {
      *weight = other.hidden_weights_1[(i, j)];
    }
    for ((i, j), bias) in
      self.hidden_biases_1.indexed_iter_mut().filter(|_| rand::gen_range(0, 1) != 0)
    {
      *bias = other.hidden_biases_1[(i, j)];
    }

    for ((i, j), weight) in
      self.hidden_weights_2.indexed_iter_mut().filter(|_| rand::gen_range(0, 1) != 0)
    {
      *weight = other.hidden_weights_2[(i, j)];
    }
    for ((i, j), bias) in
      self.hidden_biases_2.indexed_iter_mut().filter(|_| rand::gen_range(0, 1) != 0)
    {
      *bias = other.hidden_biases_2[(i, j)];
    }

    for ((i, j), weight) in
      self.hidden_weights_3.indexed_iter_mut().filter(|_| rand::gen_range(0, 1) != 0)
    {
      *weight = other.hidden_weights_3[(i, j)];
    }
    for ((i, j), bias) in
      self.hidden_biases_3.indexed_iter_mut().filter(|_| rand::gen_range(0, 1) != 0)
    {
      *bias = other.hidden_biases_3[(i, j)];
    }

    for ((i, j), weight) in
      self.hidden_weights_4.indexed_iter_mut().filter(|_| rand::gen_range(0, 1) != 0)
    {
      *weight = other.hidden_weights_4[(i, j)];
    }
    for ((i, j), bias) in
      self.hidden_biases_3.indexed_iter_mut().filter(|_| rand::gen_range(0, 1) != 0)
    {
      *bias = other.hidden_biases_4[(i, j)];
    }

    for ((i, j), weight) in
      self.output_weights.indexed_iter_mut().filter(|_| rand::gen_range(0, 1) != 0)
    {
      *weight = other.output_weights[(i, j)];
    }
    for ((i, j), bias) in
      self.output_biases.indexed_iter_mut().filter(|_| rand::gen_range(0, 1) != 0)
    {
      *bias = other.output_biases[(i, j)];
    }

    if rand::gen_range(0, 1) != 0 {
      self.a1 = other.a1;
    }
    if rand::gen_range(0, 1) != 0 {
      self.a2 = other.a2;
    }
    if rand::gen_range(0, 1) != 0 {
      self.a3 = other.a3;
    }
    if rand::gen_range(0, 1) != 0 {
      self.a4 = other.a4;
    }
  }

  fn mutate(&mut self) {
    for weight in self.hidden_weights_1.iter_mut().filter(|_| rand::gen_range(0, 1) != 0) {
      *weight = (*weight + rand::gen_range(-1.0, 1.0)).clamp(-1.0, 1.0);
    }
    for bias in self.hidden_biases_1.iter_mut().filter(|_| rand::gen_range(0, 1) != 0) {
      *bias = (*bias + rand::gen_range(-1.0, 1.0)).clamp(-1.0, 1.0);
    }

    for weight in self.hidden_weights_2.iter_mut().filter(|_| rand::gen_range(0, 1) != 0) {
      *weight = (*weight + rand::gen_range(-1.0, 1.0)).clamp(-1.0, 1.0);
    }
    for bias in self.hidden_biases_2.iter_mut().filter(|_| rand::gen_range(0, 1) != 0) {
      *bias = (*bias + rand::gen_range(-1.0, 1.0)).clamp(-1.0, 1.0);
    }

    for weight in self.hidden_weights_3.iter_mut().filter(|_| rand::gen_range(0, 1) != 0) {
      *weight = (*weight + rand::gen_range(-1.0, 1.0)).clamp(-1.0, 1.0);
    }
    for bias in self.hidden_biases_3.iter_mut().filter(|_| rand::gen_range(0, 1) != 0) {
      *bias = (*bias + rand::gen_range(-1.0, 1.0)).clamp(-1.0, 1.0);
    }

    for weight in self.hidden_weights_4.iter_mut().filter(|_| rand::gen_range(0, 1) != 0) {
      *weight = (*weight + rand::gen_range(-1.0, 1.0)).clamp(-1.0, 1.0);
    }
    for bias in self.hidden_biases_4.iter_mut().filter(|_| rand::gen_range(0, 1) != 0) {
      *bias = (*bias + rand::gen_range(-1.0, 1.0)).clamp(-1.0, 1.0);
    }

    for weight in self.output_weights.iter_mut().filter(|_| rand::gen_range(0, 1) != 0) {
      *weight = (*weight + rand::gen_range(-1.0, 1.0)).clamp(-1.0, 1.0);
    }
    for bias in self.output_biases.iter_mut().filter(|_| rand::gen_range(0, 1) != 0) {
      *bias = (*bias + rand::gen_range(-1.0, 1.0)).clamp(-1.0, 1.0);
    }

    if rand::gen_range(0, 1) != 0 {
      self.a1 += rand::gen_range(-1.0, 1.0);
    }
    if rand::gen_range(0, 1) != 0 {
      self.a2 += rand::gen_range(-1.0, 1.0);
    }
    if rand::gen_range(0, 1) != 0 {
      self.a3 += rand::gen_range(-1.0, 1.0);
    }
    if rand::gen_range(0, 1) != 0 {
      self.a4 += rand::gen_range(-1.0, 1.0);
    }
  }

  pub fn query(&self, inputs: [f32; INPUT_NEURONS]) -> usize {
    let inputs = Array2::from_shape_vec((INPUT_NEURONS, 1), inputs.to_vec()).unwrap();

    let hidden_1 = (self.hidden_weights_1.dot(&inputs) + &self.hidden_biases_1)
      .mapv(|x| Mind::prelu(x, self.a1));
    let hidden_2 = (self.hidden_weights_2.dot(&hidden_1) + &self.hidden_biases_2)
      .mapv(|x| Mind::prelu(x, self.a2));
    let hidden_3 = (self.hidden_weights_3.dot(&hidden_2) + &self.hidden_biases_3)
      .mapv(|x| Mind::prelu(x, self.a3));
    let hidden_4 = (self.hidden_weights_4.dot(&hidden_3) + &self.hidden_biases_4)
      .mapv(|x| Mind::prelu(x, self.a4));

    let output = self.output_weights.dot(&hidden_4) + &self.output_biases;
    let output_sum = output.sum();
    let output = output.mapv(|x| Mind::softmax(x, output_sum));
    output.argmax().unwrap().0
  }

  fn prelu(x: f32, a: f32) -> f32 {
    if x > 0.0 {
      x
    } else {
      a * x
    }
  }

  fn softmax(x: f32, sum: f32) -> f32 { x / sum }
}

#[derive(Clone)]
struct Scene {
  mind: Mind,
  paddle: Paddle,
  ball: Ball,
  score: usize,
  dead: bool,
}

impl Scene {
  pub fn new() -> Scene {
    let velocity = vec![-1.0, 1.0];
    let main_ball_velocity = vec2(*velocity.choose().unwrap(), *velocity.choose().unwrap());
    Scene {
      mind: Mind::new(),
      paddle: Paddle::new(vec2(
        (screen_width() - PADDLE_SIZE.x) / 2.0,
        screen_height() - PADDLE_SIZE.y - 96.0,
      )),
      ball: Ball::new(vec2(screen_width(), screen_height()) / 2.0, main_ball_velocity),
      score: 0,
      dead: false,
    }
  }

  pub fn update(&mut self) {
    if !self.dead {
      let move_x = self.mind.query([
        self.ball.velocity.x,
        self.ball.circle.y,
        self.ball.circle.point().x - self.paddle.rect.center().x,
      ]) as f32
        - 1.0;
      self.paddle.move_x(move_x);

      self.ball.move_x();
      self.ball.move_y();

      if self.ball.circle.overlaps_rect(&self.paddle.rect) {
        self.score += 1;
        self.ball.flip_y();
        while self.ball.circle.overlaps_rect(&self.paddle.rect) {
          self.ball.move_y();
        }
      }
      if self.ball.circle.y - BALL_RADIUS <= 0.0 {
        self.ball.flip_y();
      }
      if self.ball.circle.y + BALL_RADIUS >= screen_height() {
        self.dead = true;
        self.ball.flip_y();
      }
      if self.ball.circle.x - BALL_RADIUS <= 0.0
        || self.ball.circle.x + BALL_RADIUS >= screen_width()
      {
        self.ball.flip_x();
      }

      if (self.score + 1) % 200 == 0 {
        self.ball.velocity *= 1.1;
      }
    }
  }

  pub fn draw(&self, font: Font) {
    let s = &self.score.to_string();
    let text_measure = measure_text(s, Some(font), 64, 1.0);
    draw_text_ex(
      s,
      (screen_width() - text_measure.width) / 2.0,
      (screen_height() - text_measure.height + text_measure.offset_y) / 2.0,
      TextParams {
        font,
        font_size: 64,
        color: color_u8!(200, 200, 200, 255),
        ..Default::default()
      },
    );

    self.paddle.draw();
    self.ball.draw(GREEN);
  }

  fn fitness_score(&self) -> f32 { self.score as f32 }
}

const GENERATION_SCENES_COUNT: usize = 100;

struct Ecosystem {
  scenes: Vec<Scene>,
  generation: usize,
  selected_scene: usize,
  auto_change: bool,
  highscore: usize,
}

impl Ecosystem {
  pub fn new() -> Ecosystem {
    let mut scenes = Vec::with_capacity(GENERATION_SCENES_COUNT);
    for _ in 0..scenes.capacity() {
      scenes.push(Scene::new());
    }

    Ecosystem { scenes, generation: 1, selected_scene: 0, auto_change: false, highscore: 0 }
  }

  pub fn update(&mut self) {
    let mut atleast_one_alive = false;
    for scene in &mut self.scenes {
      scene.update();
      if !scene.dead {
        atleast_one_alive = true;
      }
      if scene.score > self.highscore {
        self.highscore = scene.score;
      }
    }

    if is_key_pressed(KeyCode::N) {
      self.auto_change = !self.auto_change;
    }
    if is_key_pressed(KeyCode::B) || (self.auto_change && self.scenes[self.selected_scene].dead) {
      for (i, scene) in self.scenes.iter().enumerate() {
        if scene.fitness_score() > self.scenes[self.selected_scene].fitness_score() {
          self.selected_scene = i;
        }
      }
    }

    if is_key_pressed(KeyCode::W) {
      *SPEED_UP.lock().unwrap() += 0.5;
    }
    if is_key_pressed(KeyCode::S) {
      *SPEED_UP.lock().unwrap() -= 0.5;
    }

    if !atleast_one_alive {
      self.run_ga();
      self.generation += 1;
      self.selected_scene = 0;
    }
  }

  pub fn draw(&self, font: Font) {
    let scene = &self.scenes[self.selected_scene];
    scene.draw(font);

    let s = self.generation.to_string();
    let text_measure_1 = measure_text(&s, Some(font), 32, 1.0);
    draw_text_ex(
      &s,
      10.0,
      10.0 + text_measure_1.height,
      TextParams { font, font_size: 32, color: BLACK, ..Default::default() },
    );

    let s = (self.selected_scene + 1).to_string();
    let text_measure_2 = measure_text(&s, Some(font), 32, 1.0);
    draw_text_ex(
      &s,
      10.0 * 3.0 + text_measure_1.width,
      10.0 + text_measure_2.height,
      TextParams { font, font_size: 32, color: BLACK, ..Default::default() },
    );

    let s = self.highscore.to_string();
    let text_measure_3 = measure_text(&s, Some(font), 32, 1.0);
    draw_text_ex(
      &s,
      10.0,
      10.0 * 2.0 + text_measure_1.height + text_measure_3.height,
      TextParams { font, font_size: 32, color: BLACK, ..Default::default() },
    );

    let s = SPEED_UP.lock().unwrap().to_string();
    let text_measure_4 = measure_text(&s, Some(font), 32, 1.0);
    draw_text_ex(
      &s,
      10.0 * 3.0 + text_measure_3.width,
      10.0 * 2.0 + text_measure_1.height + text_measure_4.height,
      TextParams { font, font_size: 32, color: BLACK, ..Default::default() },
    );

    if scene.dead {
      draw_rectangle(0.0, 0.0, screen_width(), screen_height(), color_u8!(255, 0, 0, 100));
    }
  }

  fn run_ga(&mut self) {
    // Selection
    let mut selected = Vec::with_capacity(GENERATION_SCENES_COUNT);

    let mut fitness_score_sum = 0.0;
    for scene in &self.scenes {
      fitness_score_sum += scene.fitness_score();
    }

    while selected.len() < GENERATION_SCENES_COUNT {
      let scene = self.scenes.choose().unwrap();
      let propability = scene.fitness_score() / fitness_score_sum;
      if rand::gen_range(0.0, 1.0) <= propability {
        selected.push(Scene { mind: scene.mind.clone(), ..Scene::new() });
      }
    }

    // Crossover
    let mut crossover = selected;
    for parents in crossover.chunks_mut(2) {
      let a = parents[0].mind.clone();
      let b = &mut parents[1].mind;
      b.breed(&a);
    }

    // Mutation
    let mut mutation = crossover;
    for scene in &mut mutation {
      scene.mind.mutate();
    }

    self.scenes = mutation;
  }
}

#[macroquad::main(winc)]
async fn main() {
  rand::srand(macroquad::miniquad::date::now() as u64);

  let font = load_ttf_font("res/font.ttf").await.unwrap();
  let mut generation = Ecosystem::new();

  loop {
    clear_background(WHITE);

    generation.update();
    generation.draw(font);

    next_frame().await;
  }
}
