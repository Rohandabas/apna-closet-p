const navbar = document.querySelector('.navbar');
const headerHeadline = document.querySelector('.header-headline');
const headerSubtitle = document.querySelector('.header-subtitle');
const cta = document.querySelector('.cta');
const form = document.querySelector('form');
const productInfo = document.querySelector('.product-info');
const productCards = document.querySelectorAll('.product-card');

const animateElements = () => {
    TweenMax.from(navbar, 1, {
        delay: .3,
        x: -40,
        opacity: 0,
        ease: Expo.easeInOut
    });
    TweenMax.from(headerHeadline, 2, {
        delay: .5,
        y: 80,
        opacity: 0,
        ease: Expo.easeInOut
    });
    TweenMax.from(headerSubtitle, 3, {
        delay: .5,
        y: 20,
        opacity: 0,
        ease: Expo.easeInOut
    });
    TweenMax.from(cta, 4, {
        delay: .5,
        y: 20,
        opacity: 0,
        ease: Expo.easeInOut
    });
    TweenMax.from(form, 5, {
        delay: 0.3,
        y: 80,
        opacity: 0,
        ease: Expo.easeInOut
    });
    TweenMax.from(productInfo, 6, {
        delay: 0.5,
        x: -100,
        opacity: 0,
        ease: Expo.easeInOut
    });
    productCards.forEach((card, index) => {
        TweenMax.from(card, 0.5, {
            delay: 0.5 + (index * 0.1),
            y: 200,
            opacity: 0,
            ease: Expo.easeInOut
        });
    });
};

animateElements();
