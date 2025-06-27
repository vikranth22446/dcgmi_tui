use clap::Parser;
use std::collections::VecDeque;
use std::fs::File;
use std::io::{BufRead, BufReader, Write};
use std::process::{Command, Stdio};
use std::sync::mpsc::{self, Sender};
use std::thread;
use std::time::{Duration, Instant};

use crossterm::event::{self, Event, KeyCode};
use crossterm::execute;
use crossterm::terminal::{disable_raw_mode, enable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen};
use ratatui::backend::CrosstermBackend;
use ratatui::layout::{Constraint, Direction, Layout};
use ratatui::style::{Color, Modifier, Style};
use ratatui::symbols::bar::Set;
use ratatui::text::{Line, Span};
use ratatui::widgets::{BarChart, Block, Borders, Paragraph};
use ratatui::Terminal;

type MetricBuffer = VecDeque<f64>;

const METRIC_NAMES: [&str; 12] = [
    "SMACT", "SMOCC", "TENSO", "FP64A", "FP32A", "FP16A", "DRAMA", "PCITX", "PCIRX", "NVLTX", "NVLRX", "FB_USED"
];

/// GPU DCGM TUI Viewer
#[derive(Parser)]
struct Args {
    /// Sampling interval in milliseconds
    #[arg(short = 'i', long = "interval", default_value_t = 100)]
    interval_ms: u64,

    /// Path to CSV log file (optional)
    #[arg(short = 'l', long = "log")]
    log_file: Option<String>,
}

fn format_bytes_with_unit(value: f64, per_sec: bool) -> String {
    const KB: f64 = 1024.0;
    const MB: f64 = KB * 1024.0;
    const GB: f64 = MB * 1024.0;
    const TB: f64 = GB * 1024.0;

    let (num, unit) = if value >= TB {
        (value / TB, "TB")
    } else if value >= GB {
        (value / GB, "GB")
    } else if value >= MB {
        (value / MB, "MB")
    } else if value >= KB {
        (value / KB, "KB")
    } else {
        (value, "B")
    };

    if unit == "B" {
        if per_sec {
            format!("{:.0} B/s", num)
        } else {
            format!("{:.0} B", num)
        }
    } else {
        if per_sec {
            format!("{:.2} {}/s", num, unit)
        } else {
            format!("{:.2} {}", num, unit)
        }
    }
}

fn percentile(sorted: &[f64], pct: usize) -> f64 {
    if sorted.is_empty() {
        return 0.0;
    }
    let rank = (pct as f64 / 100.0) * (sorted.len() - 1) as f64;
    let low = rank.floor() as usize;
    let high = rank.ceil() as usize;
    if low == high {
        sorted[low]
    } else {
        let weight = rank - low as f64;
        sorted[low] * (1.0 - weight) + sorted[high] * weight
    }
}

const CUSTOM_SET: Set = Set {
    empty: " ",
    one_eighth: "▁",
    one_quarter: "▂",
    three_eighths: "▃",
    half: "▄",
    five_eighths: "▅",
    three_quarters: "▆",
    seven_eighths: "▇",
    full: "█",
};

fn parse_metric_line(line: &str) -> Option<Vec<f64>> {
    if !line.starts_with("GPU 0") {
        return None;
    }
    let parts: Vec<&str> = line.split_whitespace().skip(1).collect();
    if parts.len() != 13 {
        return None;
    }
    let values: Vec<f64> = parts.iter().filter_map(|s| s.parse().ok()).collect();
    if values.len() == 13 {
        Some(values.into_iter().skip(1).collect())
    } else {
        None
    }
}

fn spawn_logger_thread(path: String) -> Sender<Vec<f64>> {
    let (tx, rx) = mpsc::channel::<Vec<f64>>();
    thread::spawn(move || {
        let mut file = File::create(path).expect("Failed to open log file");
        writeln!(file, "timestamp,{}", METRIC_NAMES.join(",")).ok();
        while let Ok(values) = rx.recv() {
            let timestamp = chrono::Local::now().to_rfc3339();
            let line = format!("{},{}", timestamp, values.iter().map(|v| v.to_string()).collect::<Vec<_>>().join(","));
            writeln!(file, "{}", line).ok();
        }
    });
    tx
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();
    let delay = Duration::from_millis(args.interval_ms);
    let logger: Option<Sender<Vec<f64>>> = args.log_file.map(spawn_logger_thread);

    enable_raw_mode()?;
    let mut stdout = std::io::stdout();
    execute!(stdout, EnterAlternateScreen)?;
    let backend = CrosstermBackend::new(stdout);
    let mut terminal = Terminal::new(backend)?;

    let mut child = Command::new("dcgmi")
        .arg("dmon")
        .arg("-e")
        .arg("1002,1003,1004,1006,1007,1008,1005,1009,1010,1011,1012,252")
        .arg("--entity-id").arg("0")
        .arg("-d").arg(args.interval_ms.to_string())
        .stdout(Stdio::piped())
        .spawn()?;

    let stdout = child.stdout.take().unwrap();
    let reader = BufReader::new(stdout);
    let mut lines = reader.lines();

    const HISTORY_LEN: usize = 100;
    let mut history: Vec<MetricBuffer> = vec![VecDeque::with_capacity(HISTORY_LEN); 12];
    let mut last_tick = Instant::now();

    loop {
        while let Some(Ok(line)) = lines.next() {
            if let Some(vals) = parse_metric_line(&line) {
                for (i, val) in vals.iter().enumerate() {
                    let buf = &mut history[i];
                    if buf.len() >= HISTORY_LEN {
                        buf.pop_front();
                    }
                    buf.push_back(*val);
                }

                if let Some(ref tx) = logger {
                    tx.send(vals).ok();
                }
            }
            break;
        }

        if last_tick.elapsed() >= delay {
            terminal.draw(|f| {
                let size = f.size();
                let layout = Layout::default()
                    .direction(Direction::Vertical)
                    .margin(1)
                    .constraints(METRIC_NAMES.iter().map(|_| Constraint::Length(3)).collect::<Vec<_>>())
                    .split(size);

                for (i, name) in METRIC_NAMES.iter().enumerate() {
                    let labels: Vec<String> = history[i].iter().enumerate().map(|(j, _)| j.to_string()).collect();
                    let bar_data: Vec<(&str, u64)> = labels.iter().zip(history[i].iter()).map(|(label, val)| {
                        let scaled = if *val <= 0.0 { 0.0 } else { val.sqrt() };
                        (label.as_str(), (scaled * 100.0) as u64)
                    }).collect();

                    let mut sorted: Vec<f64> = history[i].iter().copied().filter(|v| *v > 0.0).collect();
                    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
                    let (p50, p90, _p99) = if sorted.is_empty() {
                        (0.0, 0.0, 0.0)
                    } else {
                        (
                            percentile(&sorted, 50),
                            percentile(&sorted, 90),
                            percentile(&sorted, 99),
                        )
                    };

                    let chunks = Layout::default()
                        .direction(Direction::Horizontal)
                        .constraints([Constraint::Percentage(70), Constraint::Percentage(30)])
                        .split(layout[i]);

                    let barchart = BarChart::default()
                        .block(Block::default().borders(Borders::ALL).title(*name))
                        .data(&bar_data)
                        .bar_width(1)
                        .bar_style(Style::default().fg(Color::LightGreen))
                        .value_style(Style::default().fg(Color::White).add_modifier(Modifier::BOLD))
                        .bar_set(CUSTOM_SET);
                    f.render_widget(barchart, chunks[0]);

                    let stats = if *name == "PCITX" || *name == "PCIRX" || *name == "NVLTX" || *name == "NVLRX" {
                        Paragraph::new(vec![
                            Line::from(Span::raw(format!("p50: {},p90: {}", format_bytes_with_unit(p50, true), format_bytes_with_unit(p90, true)))),
                        ])
                        .block(Block::default().borders(Borders::ALL))
                        .style(Style::default().fg(Color::Gray))
                    } else if *name == "FB_USED" {
                        Paragraph::new(vec![
                            // By default MB
                            Line::from(Span::raw(format!("p50: {},p90: {}", format_bytes_with_unit(p50 * 1024.0 * 1024.0, false), format_bytes_with_unit(p90 *  1024.0 * 1024.0, false)))),
                        ])
                        .block(Block::default().borders(Borders::ALL))
                        .style(Style::default().fg(Color::Gray))
                    } else {
                        Paragraph::new(vec![
                            Line::from(Span::raw(format!("p50: {:.1}% p90: {:.1}%", p50 * 100.0, p90 * 100.0))),
                        ])
                        .block(Block::default().borders(Borders::ALL))
                        .style(Style::default().fg(Color::Gray))
                    };

                    f.render_widget(stats, chunks[1]);
                }
            })?;
            last_tick = Instant::now();
        }

        if event::poll(Duration::from_millis(10))? {
            if let Event::Key(key) = event::read()? {
                if key.code == KeyCode::Char('q') {
                    break;
                }
            }
        }
    }

    disable_raw_mode()?;
    execute!(terminal.backend_mut(), LeaveAlternateScreen)?;
    terminal.show_cursor()?;
    Ok(())
}
