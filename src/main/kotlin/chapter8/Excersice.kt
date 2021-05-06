package chapter8

import common.Result
import common.map2

fun main() {
    val list = List(Result.failure("fail"), Result(), Result(1))
    println(sequence_2(list))

    val list2 = List(Result.failure("fail"), Result(), Result(1))
    println(sequence_3(list2))

    val list3 = List(Result(3), Result(2), Result(1))
    println(sequence_2(list3))
}

// 8-5
fun <A> flattenResult_1(list: List<Result<A>>): List<A> =
    list.foldLeft(List()) { acc: List<A> ->
        { result: Result<A> ->
            acc.concat(result.flatMap { Result(List(it)) }.getOrElse(List()))
        }
    }

// 1. Result<A> -> List<A> 로 만들고 flatten
fun <A> flattenResult_2(list: List<Result<A>>): List<A> =
    list.flatMap { it.map { ra -> List(ra) }.getOrElse(List()) }

// 8-6
fun <A> sequence_1(list: List<Result<A>>): Result<List<A>> {
    val successResults = flattenResult_2(list)
    return if (successResults.isEmpty()) Result()
    else Result(successResults)
}

fun <A> sequence_2(list: List<Result<A>>): Result<List<A>> =
    list.foldRight(Result(List())) { x: Result<A> ->
        { y: Result<List<A>> ->
            map2(
                x,
                y
            ) { a: A -> { b: List<A> -> b.cons(a) } }
        }
    }

fun <A> sequence_3(list: List<Result<A>>): Result<List<A>> =
    list.foldLeft(Result(List())) { x: Result<List<A>> ->
        { y: Result<A> ->
            map2(x, y) { a: List<A> ->
                { b: A -> a.cons(b) }
            }
        }
    }

// 8-7
fun <A, B> traverse(list: List<A>, f: (A) -> Result<B>): Result<List<B>> = list.foldLeft(Result(List())) { x ->
    { y -> map2(x, f(y)) { a -> { b -> a.cons(b) } } }
}

fun <A> sequenceWithTraverse(list: List<Result<A>>): Result<List<A>> = traverse(list) { it } // ??????

// 8-8
fun <A, B, C> zipWith(list1: List<A>, list2: List<B>, f: (A) -> (B) -> C): List<C> {
    fun zipWith_(remainList1: List<A>, remainList2: List<B>, acc: List<C>): List<C> = when (remainList1) {
        is List.Nil -> acc
        is List.Cons -> when (remainList2) {
            is List.Nil -> acc
            is List.Cons -> zipWith_(
                remainList1.tail,
                remainList2.tail,
                acc.cons(f(remainList1.head)(remainList2.head))
            )
        }
    }
    return zipWith_(list1, list2, List())
}

// 8-9
fun conbination_1(l1: List<String>, l2: List<String>): List<String> = l1.flatMap { ll1 -> l2.map { ll2 -> ll1 + ll2 } }
fun <A, B, C> conbination_2(l1: List<A>, l2: List<B>, f: (A) -> (B) -> C): List<C> =
    l1.flatMap { ll1 -> l2.map { ll2 -> f(ll1)(ll2) } }

// 8-10
fun <A, B> unzip(list: List<Pair<A, B>>): Pair<List<A>, List<B>> = list.coFoldRight(Pair(List(), List())) { elem ->
    { p ->
        Pair(p.first.cons(elem.first), p.second.cons(elem.second))
    }
}

sealed class List<out A> {

    abstract fun isEmpty(): Boolean

    abstract fun init(): List<A>

    // 8-1
    abstract fun lengthMemoized(): Int

    // 8-2
    abstract fun headSafe(): Result<A>

    // 8-3 (첨부터 끝까지 원소 바뀌치기)
    fun lastSafe(): Result<A> = this.foldLeft(Result()) { { a -> Result(a) } }

    // 8-4 (마지막 원소를 들고 그냥 순회할 뿐)
    fun headSafeInefficient(): Result<A> = foldRight(Result()) { x -> { _ -> Result(x) } }

    // 8-11
    fun <A1, A2> unzip2(f: (A) -> Pair<A1, A2>): Pair<List<A1>, List<A2>> =
        this.coFoldRight(Pair(invoke(), invoke())) { elem ->
            { p ->
                f(elem).let {
                    Pair(
                        p.first.cons(it.first),
                        p.second.cons(it.second)
                    )
                }
            }
        }

    fun setHead(a: @UnsafeVariance A): List<A> = when (this) {
        Nil -> throw IllegalStateException("setHead called on an empty list")
        is Cons -> Cons(a, this.tail)
    }

    fun cons(a: @UnsafeVariance A): List<A> = Cons(a, this)

    fun concat(list: List<@UnsafeVariance A>): List<A> = concat(this, list)

    fun concatViaFoldRight(list: List<@UnsafeVariance A>): List<A> = concatViaFoldRight(this, list)

    fun drop(n: Int): List<A> = drop(this, n)

    fun dropWhile(p: (A) -> Boolean): List<A> = dropWhile(this, p)

    fun reverse(): List<A> = foldLeft(Nil as List<A>) { acc -> { acc.cons(it) } }

    fun <B> foldRight(identity: B, f: (A) -> (B) -> B): B = foldRight(this, identity, f)

    fun <B> foldLeft(identity: B, f: (B) -> (A) -> B): B = foldLeft(identity, this, f)

    fun length(): Int = foldLeft(0) { { _ -> it + 1 } }

    fun <B> foldRightViaFoldLeft(identity: B, f: (A) -> (B) -> B): B =
        this.reverse().foldLeft(identity) { x -> { y -> f(y)(x) } }

    fun <B> coFoldRight(identity: B, f: (A) -> (B) -> B): B = coFoldRight(identity, this.reverse(), identity, f)

    fun <B> map(f: (A) -> B): List<B> = foldLeft(Nil) { acc: List<B> -> { h: A -> Cons(f(h), acc) } }.reverse()

    fun <B> flatMap(f: (A) -> List<B>): List<B> = flatten(map(f))

    fun filter(p: (A) -> Boolean): List<A> = flatMap { a -> if (p(a)) List(a) else Nil }

    internal object Nil : List<Nothing>() {
        override fun lengthMemoized(): Int = 0
        override fun init(): List<Nothing> = throw IllegalStateException("init called on an empty list")
        override fun isEmpty() = true
        override fun toString(): String = "[NIL]"
        override fun headSafe(): Result<Nothing> = Result()
    }

    internal class Cons<out A>(
        internal val head: A,
        internal val tail: List<A>
    ) : List<A>() {
        private val length: Int = tail.lengthMemoized() + 1
        override fun lengthMemoized() = length
        override fun init(): List<A> = reverse().drop(1).reverse()
        override fun isEmpty() = false
        override fun toString(): String = "[${toString("", this)}NIL]"
        private tailrec fun toString(acc: String, list: List<A>): String = when (list) {
            Nil -> acc
            is Cons -> toString("$acc${list.head}, ", list.tail)
        }

        override fun headSafe(): Result<A> = Result(head)
    }

    companion object {
        fun <A> cons(a: A, list: List<A>): List<A> = Cons(a, list)
        tailrec fun <A> drop(list: List<A>, n: Int): List<A> = when (list) {
            Nil -> list
            is Cons -> if (n <= 0) list else drop(list.tail, n - 1)
        }

        tailrec fun <A> dropWhile(list: List<A>, p: (A) -> Boolean): List<A> = when (list) {
            Nil -> list
            is Cons -> if (p(list.head)) dropWhile(list.tail, p) else list
        }

        fun <A> concat(list1: List<A>, list2: List<A>): List<A> = list1.reverse().foldLeft(list2) { x -> x::cons }
        fun <A> concatViaFoldRight(list1: List<A>, list2: List<A>): List<A> =
            foldRight(list1, list2) { x -> { y -> Cons(x, y) } }

        fun <A, B> foldRight(list: List<A>, identity: B, f: (A) -> (B) -> B): B =
            when (list) {
                Nil -> identity
                is Cons -> f(list.head)(foldRight(list.tail, identity, f))
            }

        tailrec fun <A, B> foldLeft(acc: B, list: List<A>, f: (B) -> (A) -> B): B =
            when (list) {
                Nil -> acc
                is Cons -> foldLeft(f(acc)(list.head), list.tail, f)
            }

        tailrec fun <A, B> coFoldRight(acc: B, list: List<A>, identity: B, f: (A) -> (B) -> B): B =
            when (list) {
                Nil -> acc
                is Cons -> coFoldRight(f(list.head)(acc), list.tail, identity, f)
            }

        fun <A> flatten(list: List<List<A>>): List<A> = list.coFoldRight(Nil) { x -> x::concat }
        operator fun <A> invoke(vararg az: A): List<A> =
            az.foldRight(Nil) { a: A, list: List<A> -> Cons(a, list) }
    }
}

fun sum(list: List<Int>): Int = list.foldRight(0) { x -> { y -> x + y } }

fun product(list: List<Double>): Double = list.foldRight(1.0) { x -> { y -> x * y } }

fun triple(list: List<Int>): List<Int> =
    List.foldRight(list, List()) { h -> { t: List<Int> -> t.cons(h * 3) } }

fun doubleToString(list: List<Double>): List<String> =
    List.foldRight(list, List()) { h -> { t: List<String> -> t.cons(h.toString()) } }