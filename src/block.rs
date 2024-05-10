use std::{
    future::Future,
    sync::{Arc, Condvar, Mutex},
    task::{Context, Poll, Wake, Waker},
};

enum State {
    Empty,
    Waiting,
    Notified,
}

struct Signal {
    state: Mutex<State>,
    cond: Condvar,
}

impl Signal {
    fn new() -> Self {
        Self { state: Mutex::new(State::Empty), cond: Condvar::new() }
    }

    fn wait(&self) {
        let mut state = self.state.lock().unwrap();
        match *state {
            State::Notified => {
                *state = State::Empty;
                return;
            }
            State::Waiting => {
                unreachable!("Multiple threads waiting on the same signal");
            }
            State::Empty => {
                *state = State::Waiting;
                while let State::Waiting = *state {
                    state = self.cond.wait(state).unwrap();
                }
            }
        }
    }

    fn notify(&self) {
        let mut state = self.state.lock().unwrap();
        match *state {
            State::Empty => *state = State::Notified,
            State::Waiting => {
                *state = State::Empty;
                self.cond.notify_one();
            }
            _ => {}
        }
    }
}

impl Wake for Signal {
    fn wake(self: Arc<Self>) {
        self.notify();
    }
}

pub fn block_on<F: Future>(mut fut: F) -> F::Output {
    let mut fut = unsafe { std::pin::Pin::new_unchecked(&mut fut) };

    let signal = Arc::new(Signal::new());
    let waker = Waker::from(Arc::clone(&signal));
    let mut context = Context::from_waker(&waker);

    loop {
        match fut.as_mut().poll(&mut context) {
            Poll::Pending => signal.wait(),
            Poll::Ready(item) => break item,
        }
    }
}
